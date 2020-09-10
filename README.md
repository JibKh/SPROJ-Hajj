# NVIDIA-FlowNet2-Google-Colab
This is the Google Colab Notebook implementation of the NVIDIA-flownet2-pytorch at https://github.com/NVIDIA/flownet2-pytorch.

## How to run
1) Enable GPU from 'Runtime'
2) Run the first cell and check which GPU is running. Tesla P100-PCIE or Tesla T4 work currently. Tesla T4 may give an error THCudaCheckError, however it will work. Tesla K80 will not work.
3) If those are not the GPU being used, then go to 'Runtime' > 'Factory Reset Runtime'. Repeat until you get either of the GPU.
