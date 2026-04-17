# Grid
- 每次kernel启动一定会启动一个Grid
- 每个Grid是由一个或多个Block组成的
    - Grid可以由1D, 2D，3D的block组成
    - 代码中可以通过GridDim.x, GridDim.y, GridDim.z来获取Grid中各维度的block数量
- 一个Grid中的总thread个数可以超过硬件的总thread数目
    - 因为Grid是以block为单位进行调度的，一个block中的所有threads是并行的，但block之间不一定是并行的。
    - 各个block可以是一批一批执行的，并不是所有block并行，所以在Grid在代码中分配的thread个数可以超过硬件中实际的thread执行单元数量

# Block
- 每个Block由一个或多个Thread组成的，是一个线程组，一组线程共同完成对一小块数据的计算
    - block可以由1D, 2D，3D的thread组成
    - 代码中可以通过BlockDim.x, BlockDim.y, BlockDim.z来获取Block中各维度的thread数量
    - 一个block中的所有threads是并行的，但block之间不一定是并行的
    - 一个block中的线程数量存在个数限制，通常是1024（取决于GPU架构），同时还受各维度（x/y/z）数量限制
- 每个block通过BlockIdx.x, BlockIdx.y, BlockIdx.z来获取在当前Grid中的位置索引
- 一个block只能在一个SM上执行，一个SM上可以同时驻留多个block
- 每个block内的thread共享一块shared memory
    - 用于同block不同thread之间的信息交互
    - 不同block之间的shared memory不共享
    - block中shared memory的使用会影响一个SM中同时驻留的block数量

# Thread
- 每个Thread负责一个或多个元素的具体计算，是cuda中的最小计算单元
- Thread的索引与坐标
    - 每个thread，通过ThreadIdx.x, ThreadIdx.y, ThreadIdx.z来获取在当前block中的位置索引
    - 每个thread通过以下公式来计算在全局数据中的索引:
      `int idx = blockIdx.x * blockDim.x + threadIdx.x;` (对于1D情况)
      或者在2D情况下:
      `int idx = blockIdx.x * blockDim.x + threadIdx.x;
       int idy = blockIdx.y * blockDim.y + threadIdx.y;
       int globalIdx = idy * gridDim.x * blockDim.x + idx;`
    - 通过这些位置索引，thread得以知晓自己应该处理哪一部分数据
- 每32个Thread组成一个warp，warp是GPU的基本调度单元
    - 每个warp中的thread执行相同的指令，但处理不同的数据
- 每个thread私有一部分寄存器资源，若分配过多可能影响SM中容纳的thread/block数量

# 层级与关系
- Grid -> Block -> Thread
- 如果以向量加做比喻，那么Grid可以理解为一整个向量，Block可以理解为一段向量数据，Thread可以理解为向量元素
- 1D, 2D, 3D的组织结构往往和需要处理的数据相对应。
    - 若处理一个向量，则常用1D组织结构
    - 若处理一个图像，Grid往往是一个2D block组成，每个block处理一部分图像，block由2D thread组成，每个thread对应一个像素
    - 三维度的组织结构，方便程序员根据不同的实际数据（向量。图像，多维数据）进行具体的数据映射
- Grid往往用来控制问题规模，数据量大，block的逻辑个数就越多
- Block往往用来进行性能调优，根据硬件特性不同，使用不同大小的block，以最大限度发挥硬件效能
- 与for循环相比，Grid就是循环整体，thread对应每一次循环体的执行，block对应循环中的一部分，决定如何分组更方便GPU进行执行