# Hajj Project for SPROJ
This is our SPROJ code submission for the Hajj Project. <br>
Crowd Control System for Hajj
• Creating a system which uses machine learning to control large crowds based on their speed, direction, and density.
• The system will detect and inform the attendant of any danger and its possible solutions (such as crowd collisions and redirecting them).
• At the current stage, I have created a semi-automated system which can find the velocity of any object in a video without the need of an additional sensor.
• We use OpenCV on Google Colab along with the help of FlowNet2.0 and NWPU frameworks.
• This project can be scaled to various different fields such as Smart City for self-driving cars or vehicular traffic.

## How to run
1) Open the ipynb file and run the first "Check Gpu" cell. YOU MUST open it through github.
2) Tesla P100-PCIE or Tesla T4 work currently. Tesla T4 may give an error THCudaCheckError, however it will work. Tesla K80 will not work.
3) If those are not the GPU being used, then go to 'Runtime' > 'Factory Reset Runtime'. Repeat until you get either of the GPU.
4) Fill the User Input section as per your requirements
5) Go to the manual annotation section and read through that
