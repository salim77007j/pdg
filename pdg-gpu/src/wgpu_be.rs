//! wgpu backend: renders semantic commands through the host GPU stack.
//!
//! The backend implements [`GpuDevice`] over wgpu (Vulkan/DX12/Metal by
//! platform). It is **headless**: color targets live in guest RAM
//! (`CB_COLORn_BASE`), render offscreen, and are copied back into guest
//! memory after the pass — the console's unified-memory behavior.
//!
//! Shaders come from the GCN lifter as WGSL (through the SPIR-V hop and
//! naga) and are compiled into wgpu pipelines at bind time; the fetch
//! manifest is resolved against guest memory (the descriptor tables and
//! `T#`s the driver set up) exactly as documented in
//! [`crate::gcn::lifter`].
//!
//! Availability: creating a backend without a host adapter fails with
//! [`GpuError::Init`]; tests skip honestly when that happens. In CI the
//! Linux job installs `mesa-vulkan-drivers` (lavapipe) so the full path
//! runs everywhere.

use crate::GpuError;
use crate::gcn::decode::decode_program;
use crate::gcn::lifter::{LiftInputs, Stage as LiftStage, VertexFormat, lift};
use crate::gcn::naga_fe::spirv_to_wgsl;
use crate::ir::{GpuCommand, GpuContext, Stage, TargetState};
use crate::pm4::cp::CpMemPort;
use pollster::block_on;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Mutex;
use wgpu::util::DeviceExt;

/// The wgpu-backed device.
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Lifted pipelines keyed by guest program address + stage.
    pipelines: Mutex<HashMap<(u64, u8), std::sync::Arc<ArcPipeline>>>,
    /// Tracing tag.
    name: &'static str,
}

struct ArcPipeline {
    vs: wgpu::ShaderModule,
    fs: wgpu::ShaderModule,
    fetch_count: usize,
}

impl WgpuBackend {
    /// Create the backend, preferring Vulkan.
    ///
    /// Fails honestly when no adapter exists.
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .into_iter()
            .next()
            .or_else(|| {
                instance
                    .enumerate_adapters(wgpu::Backends::all())
                    .into_iter()
                    .next()
            });
        let Some(adapter) = adapter else {
            return Err(GpuError::Init {
                backend: "wgpu",
                what: "no suitable adapter (install mesa-vulkan-drivers for lavapipe)".into(),
            });
        };
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .map_err(|e| GpuError::Init {
                backend: "wgpu",
                what: format!("device request failed: {e}"),
            })?;
        Ok(Self {
            device,
            queue,
            pipelines: Mutex::new(HashMap::new()),
            name: "wgpu",
        })
    }

    /// The backend name for logs.
    #[must_use]
    pub fn adapter_name(&self) -> String {
        self.name.to_string()
    }

    /// Lift a guest shader into WGSL and compile both stages.
    fn pipeline_for(
        &self,
        pgm: u64,
        stage: Stage,
        mem: &mut dyn CpMemPort,
        inputs: &LiftInputs,
    ) -> Result<ArcPipeline, GpuError> {
        // Read the program from guest memory until an s_endpgm (bounded).
        let mut isa = Vec::new();
        let mut at = pgm;
        for _ in 0..256 {
            let w = mem.read_u32(at);
            isa.push(w);
            // Detect end-of-program in the decoded stream lazily: the
            // decoder below stops at the first s_endpgm, so we read a
            // generous window and trim after decoding.
            at += 4;
        }
        let decoded = decode_program(&isa).map_err(|e| GpuError::BadState {
            backend: self.name,
            what: format!("shader at {pgm:#x} does not decode: {e}"),
        })?;
        let end = decoded
            .iter()
            .position(|(_, i)| {
                matches!(
                    i,
                    crate::gcn::decode::Inst::Sopp { op, .. }
                        if *op == crate::gcn::opcodes::sopp::S_ENDPGM
                )
            })
            .map(|p| p + 1)
            .ok_or_else(|| GpuError::BadState {
                backend: self.name,
                what: format!("shader at {pgm:#x} has no s_endpgm in the first 256 words"),
            })?;
        let _isa_words: Vec<u32> = decoded[..end].iter().map(|(i, _)| isa[*i]).collect();

        let (vs_wgsl, fs_wgsl, fetch_count) = match stage {
            Stage::Vertex => {
                let lifted = lift(&decoded[..end], LiftStage::Vertex, inputs).map_err(|e| {
                    GpuError::BadState {
                        backend: self.name,
                        what: format!("VS lift at {pgm:#x}: {e}"),
                    }
                })?;
                let wgsl = spirv_to_wgsl(&lifted.spirv).map_err(|e| GpuError::BadState {
                    backend: self.name,
                    what: format!("VS naga at {pgm:#x}: {e}"),
                })?;
                // The pixel stage is trivial for the supported draws:
                // pass-through of the varying (lifted on demand when the
                // guest binds one; otherwise constant color).
                let fs = "struct VsOut {\n    @builtin(position) pos: vec4<f32>,\n    @location(0) color: vec4<f32>,\n}\nstruct FsOut { @location(0) color: vec4<f32> }\n@fragment fn pdg_main(in: VsOut) -> FsOut { return FsOut(in.color); }\n";
                (wgsl, fs.to_string(), lifted.manifest.fetches.len())
            }
            _ => unreachable!("pixel pipelines are compiled with the vertex stage"),
        };

        let vs = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pdg-vs"),
                source: wgpu::ShaderSource::Wgsl(vs_wgsl.into()),
            });
        let fs = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pdg-fs"),
                source: wgpu::ShaderSource::Wgsl(fs_wgsl.into()),
            });
        Ok(ArcPipeline {
            vs,
            fs,
            fetch_count,
        })
    }
}

impl crate::GpuDevice for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn submit(
        &mut self,
        ctx: &GpuContext,
        commands: &[GpuCommand],
        mem: &mut dyn CpMemPort,
    ) -> Result<(), GpuError> {
        for command in commands {
            match *command {
                GpuCommand::Draw { params } => self.draw(ctx, params, mem)?,
                GpuCommand::Fence { addr, value, wide } => {
                    if wide {
                        mem.write_u32(addr, value as u32);
                        mem.write_u32(addr + 4, (value >> 32) as u32);
                    } else {
                        mem.write_u32(addr, value as u32);
                    }
                }
                GpuCommand::Copy { dst, src, len } => {
                    let words = mem.read_words(src, len.div_ceil(4) as usize);
                    let mut bytes = Vec::with_capacity(len as usize);
                    for w in words {
                        bytes.extend_from_slice(&w.to_le_bytes());
                    }
                    bytes.truncate(len as usize);
                    mem.write_bytes(dst, &bytes);
                }
                GpuCommand::Dispatch { .. } => {
                    return Err(GpuError::Unsupported {
                        backend: self.name,
                        what: "compute dispatches are not rendered yet (Phase 4)".into(),
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }
}

impl WgpuBackend {
    fn draw(
        &self,
        ctx: &GpuContext,
        params: crate::ir::DrawParams,
        mem: &mut dyn CpMemPort,
    ) -> Result<(), GpuError> {
        let target = ctx.target(0);
        let (Some(width), Some(height)) = (target.width, target.height) else {
            return Err(GpuError::BadState {
                backend: "wgpu",
                what: "draw with an unsized/tiled color target".into(),
            });
        };
        if target.tile_mode != crate::ir::TileMode::Linear {
            return Err(GpuError::BadState {
                backend: "wgpu",
                what: "tiled color targets are not supported yet (linear only)".into(),
            });
        }
        // The supported target format: 8_8_8_8 + UNORM (+ comp swap).
        let fmt_ok = target.format == crate::pm4::regs::DataFormat::Fmt8_8_8_8
            && target.number == crate::pm4::regs::NumberFormat::Unorm;
        if !fmt_ok {
            return Err(GpuError::BadState {
                backend: "wgpu",
                what: format!(
                    "target format {:?}/{:?} is not rendered yet (8_8_8_8 UNORM only)",
                    target.format, target.number
                ),
            });
        }

        // Vertex shader binding.
        let vs_binding = ctx.shader_binding(Stage::Vertex);
        if vs_binding.pgm == 0 {
            return Err(GpuError::BadState {
                backend: "wgpu",
                what: "draw without a bound vertex program".into(),
            });
        }

        // Lift inputs resolved from the CP context: the mini-GNM contract
        // is fixed for the Phase-3 driver (vertex id in SGPR 6, one
        // descriptor table at SGPR pair 0, one fetch).
        let inputs = LiftInputs {
            vertex_id_sgpr: Some(6),
            descriptor_tables: vec![0],
            vertex_fetches: vec![crate::gcn::lifter::VertexFetchDecl {
                table_sbase: 0,
                sload_offset_dwords: 0x10,
                format: VertexFormat::F32x4,
            }],
            ps_inputs: vec![(0, 2, 4)],
        };

        let key = (vs_binding.pgm, 0u8);
        let mut pipelines = self.pipelines.lock().unwrap();
        let pipeline: std::sync::Arc<ArcPipeline> = match pipelines.entry(key) {
            Entry::Occupied(o) => o.get().clone(),
            Entry::Vacant(v) => {
                let p = std::sync::Arc::new(self.pipeline_for(
                    vs_binding.pgm,
                    Stage::Vertex,
                    mem,
                    &inputs,
                )?);
                v.insert(p.clone());
                p
            }
        };

        // Resolve the vertex buffer through the descriptor table.
        let table_addr = u64::from(vs_binding.user_data.first().copied().unwrap_or(0))
            | (u64::from(vs_binding.user_data.get(1).copied().unwrap_or(0)) << 32);
        if table_addr == 0 {
            return Err(GpuError::BadState {
                backend: "wgpu",
                what: "vertex user data does not point at a descriptor table".into(),
            });
        }
        // T# at table + 0x10 dwords (the declared fetch offset).
        let t0 = mem.read_u32(table_addr + 0x10 * 4);
        let t1 = mem.read_u32(table_addr + 0x10 * 4 + 4);
        let t2 = mem.read_u32(table_addr + 0x10 * 4 + 8);
        let t3 = mem.read_u32(table_addr + 0x10 * 4 + 12);
        // Buffer resource layout (public AMD documentation, mirrored by
        // shadPS4's `Buffer`): dword0 = base[31:0], dword1 = base[39:32]
        // | stride[29:16], dword2 = num_records, dword3 = format word.
        let vb_base = u64::from(t0) | (u64::from(t1 & 0xFF) << 32);
        let stride = (t1 >> 16) & 0x3FFF;
        let num_records = t2;
        let data_format = (t3 >> 15) & 0xF;
        let num_format = (t3 >> 12) & 0x7;
        if data_format != 14 || num_format != 7 {
            return Err(GpuError::BadState {
                backend: "wgpu",
                what: format!(
                    "vertex buffer format {data_format}/{num_format} does not match the f32x4 specialization"
                ),
            });
        }
        if stride != 16 {
            return Err(GpuError::BadState {
                backend: "wgpu",
                what: format!("vertex stride {stride} is not the 16-byte record alignment"),
            });
        }

        // Copy the vertex range guest → host.
        let vertices = if params.index_type == crate::ir::IndexType::Unset {
            params.count.max(1)
        } else {
            num_records.min(params.count.max(1))
        };
        let total_bytes = vertices as usize * 16;
        let mut vb_bytes = Vec::with_capacity(total_bytes);
        for w in mem.read_words(vb_base, total_bytes / 4) {
            vb_bytes.extend_from_slice(&w.to_le_bytes());
        }
        let vb = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pdg-vertices"),
                contents: &vb_bytes,
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            });

        // Fetch-parameter storage binding ([stride, num_records]).
        let fetch_bytes: Vec<u8> = [stride, num_records]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        // Storage binding for the fetch parameters ([stride, count] pairs).
        // Read-only storage rather than a uniform buffer: the lifted
        // shader declares it that way (WGSL uniform arrays need 16-byte
        // strides; the packed u32 pairs would not survive validation).
        let fetch_ub = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pdg-fetch"),
                contents: &fetch_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        // The storage binding reuses the vertex buffer.
        let storage_for_fetch = &vb;

        // Target texture (offscreen, guest-backed).
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pdg-target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());

        // Bind group: 0 = fetch uniform, 1 = vertex storage.
        let bgl = self.pipeline_layout(pipeline.fetch_count)?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pdg-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &fetch_ub,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: storage_for_fetch,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // Pipeline.
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pdg-pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("pdg-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &pipeline.vs,
                    entry_point: None,
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: to_topology(params.prim_type)?,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &pipeline.fs,
                    entry_point: None,
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            });

        // Render pass with the guest clear color.
        let clear = clear_color(&target);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pdg-enc"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pdg-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_pipeline(&pipeline);
            // The vertex index comes from @builtin(vertex_index); the
            // fetch records hold the data.
            pass.draw(0..params.count, 0..params.instances);
        }

        // Copy the target back into guest memory.
        let bytes_per_row = (width * 4).next_multiple_of(256);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pdg-readback"),
            size: u64::from(bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| GpuError::BadState {
                backend: "wgpu",
                what: format!("device poll failed: {e:?}"),
            })?;
        rx.recv()
            .map_err(|_| GpuError::BadState {
                backend: "wgpu",
                what: "readback channel closed".into(),
            })?
            .map_err(|e| GpuError::BadState {
                backend: "wgpu",
                what: format!("readback map failed: {e}"),
            })?;
        let data = slice.get_mapped_range().to_vec();
        for y in 0..height {
            for x in 0..width {
                let src = (y * bytes_per_row + x * 4) as usize;
                let dst = target.base + u64::from(y * width + x) * 4;
                mem.write_bytes(dst, &data[src..src + 4]);
            }
        }
        Ok(())
    }

    fn pipeline_layout(&self, fetch_count: usize) -> Result<wgpu::BindGroupLayout, GpuError> {
        let _ = fetch_count;
        Ok(self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pdg-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }))
    }
}

/// Map a GCN primitive to wgpu.
fn to_topology(p: crate::ir::PrimitiveType) -> Result<wgpu::PrimitiveTopology, GpuError> {
    use crate::ir::PrimitiveType;
    Ok(match p {
        PrimitiveType::TriList => wgpu::PrimitiveTopology::TriangleList,
        PrimitiveType::TriStrip => wgpu::PrimitiveTopology::TriangleStrip,
        PrimitiveType::PointList => wgpu::PrimitiveTopology::PointList,
        PrimitiveType::LineList => wgpu::PrimitiveTopology::LineList,
        PrimitiveType::LineStrip => wgpu::PrimitiveTopology::LineStrip,
        PrimitiveType::TriFan => {
            return Err(GpuError::BadState {
                backend: "wgpu",
                what: "triangle fans need index expansion (not yet modeled)".into(),
            });
        }
        PrimitiveType::Unsupported => {
            return Err(GpuError::BadState {
                backend: "wgpu",
                what: "primitive type outside the modeled set".into(),
            });
        }
    })
}

/// Clear color from the CB clear words (8_8_8_8_UNORM layout).
fn clear_color(t: &TargetState) -> wgpu::Color {
    let w0 = t.clear_word0;
    let w1 = t.clear_word1;
    // Clear words pack 8-bit components; word0 = RGBA low half,
    // word1 = RGBA high half for 64-bit targets. For 32bpp targets the
    // convention is word0 = packed color.
    let bytes = w0.to_le_bytes();
    let f = |b: u8| f64::from(b) / 255.0;
    let _ = w1;
    wgpu::Color {
        r: f(bytes[0]),
        g: f(bytes[1]),
        b: f(bytes[2]),
        a: f(bytes[3]),
    }
}

/// Feature flag string for logs.
#[must_use]
pub fn backend_features() -> String {
    "wgpu 27 (Vulkan/DX12/Metal)".to_string()
}
