//! MMU benchmarks: page-walk cost with and without the software TLB.

use criterion::{Criterion, criterion_group, criterion_main};
use pdg_memory::{
    AccessWidth, Mmu,
    bus::Bus,
    mmu::AccessKind,
    page_table::{PageFlags, PageTableBuilder},
};

fn fixture() -> (Bus, Mmu) {
    let mut bus = Bus::new(0x20_0000).unwrap();
    let mut builder = PageTableBuilder::new(&mut bus, 0x1000, 0x2_0000);
    builder
        .map_2m(0, 0, PageFlags::PRESENT | PageFlags::RW | PageFlags::NX)
        .unwrap();
    (bus, Mmu::new(0x1000))
}

fn bench_walk(c: &mut Criterion) {
    let mut group = c.benchmark_group("mmu");
    group.bench_function("translate_tlb_hit", |b| {
        let (bus, mut mmu) = fixture();
        b.iter(|| {
            mmu.translate(&bus, 0x1F_F000, AccessKind::Read).unwrap();
        });
    });
    group.bench_function("translate_full_walk", |b| {
        let (bus, mut mmu) = fixture();
        mmu.set_tlb_enabled(false);
        b.iter(|| {
            mmu.translate(&bus, 0x1F_F000, AccessKind::Read).unwrap();
        });
    });
    group.bench_function("read_u64_via_mmu_tlb", |b| {
        let (mut bus, mut mmu) = fixture();
        mmu.write(&mut bus, AccessWidth::W64, 0x1F_F000, 0xDEAD)
            .unwrap();
        b.iter(|| {
            mmu.read(&bus, AccessWidth::W64, 0x1F_F000).unwrap();
        });
    });
    group.finish();
}

criterion_group!(benches, bench_walk);
criterion_main!(benches);
