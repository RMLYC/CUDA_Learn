# Share memory 共享内存
- Share memory 是SM上的程序员可通过编程操作的片上内存；每个SM都有一片独有的share memory.
- Share memory 和 L1 cache 都是每个SM私有的，部分NVIDIA显卡上L1 cache和share memory在同一块片上SRAM中
    - Share memory 上的数据是可以通过软件操作并管理其声明周期的
    - L1 cache上的数据是硬件调度和分配的，无法软件管理和操作
    - 对于share memory和L1 cache在同一片SRAM上的情况，share memory的大小分配会对L1 cache产生影响
## Share memory 和 Global memory 的数据访问对比
- Shared Memory 由位于SM内部的片上SRAM实现，有更短的访问路径，更低的访问延时和更高的片上带宽。
- 对于数据复用较多的算法，通过将可复用数据暂存刀Shared memory, GPU kernel可以显著减少昂贵的片外global memory访问
- 线程访问Share memory: thread -> shared memory
- 线程访问Global memory: thread -> L1 cache or By pass -> L2 cache -> Global memory

# 分块（Tiling）是如何减少对全局内存访问的
- 以一个 `M x M` 的方阵乘为例， 一次计算会把每个数据访问M次。总共有M * M * M * 2次全局内存的访问
- 每个元素产生了M次数据复用，适合使用Shared memory暂存可复用数据，减少对全局内存的访问
- 假设每个Tile涉及大小为4x4, 每次分别加载两个4x4的Tile到share memory, 在share memory上对两个Tile做矩阵乘，以此类推，循环。
- Tile上的每次矩阵乘都只从global memory上加载一次，在Share memory上复用4次，这样每做一次完整的矩阵乘，每个元素在全局内存上的访问次数会下降为原来的四分之一，即M/4。
- 在SM的片上Share memory资源限制内，全局内存的访问次数下降为1/TILE_SIZE
- Tiling 将原本需要通过访问Global memory获得的数据，转移到了share memory上。用更高带宽的share memory访问替代了Global memory访问，减少了Global memory访问次数，提升了运算性能

