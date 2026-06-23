param(
    [int]$Blocks = 64,
    [int]$TokensPerBlock = 128,
    [int]$HeadDim = 64,
    [int]$HbmCapacityMb = 16,
    [int]$CpuCapacityMb = 128
)

python -m goldenexperience.benchmarks.synthetic `
    --blocks $Blocks `
    --tokens-per-block $TokensPerBlock `
    --head-dim $HeadDim `
    --hbm-capacity-mb $HbmCapacityMb `
    --cpu-capacity-mb $CpuCapacityMb

