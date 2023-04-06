## Pool Shot Predictor

This is a contest sponsored by Nvidia and presented by [CVZone](https://www.computervision.zone/). 

The task is to predict the path of the ball being hit and to predict whether it will go in the pocket or not 
having to predict before the cue ball touches the colored ball.

To solve the problem, the contour approach was used, finding contours of Stick, Cue and Ball in specific frame, then making math calculations and 
projections it is able to tell if the ball will fall inside one of the 6 holes in the table. 
If that is the case, draw in green the path and the ghost balls else will draw in red. 

For finding contours some tweaks in Canny variables and Kernel size was made to get the all contours, 
then are filtered to keep only the ones of interest. 

Holes and lower edge will have the same coordinates in every case.

### Two example of the solution: 

<img width="500" alt="Screen Shot 2023-04-05 at 20 50 26 2" src="https://user-images.githubusercontent.com/5184731/230246648-024e7502-5d8f-4161-bc91-91a765302e60.png">
<img width="500" alt="Screen Shot 2023-04-05 at 20 50 18 2" src="https://user-images.githubusercontent.com/5184731/230246664-c86ca92d-bb9a-4092-926e-1d22af6f76cc.png">

### The complete solution can be seen [here](https://www.youtube.com/watch?v=UAJPBr9GCbU)

### Original video to run this project can be found [here](https://usercontent.one/wp/www.computervision.zone/wp-content/uploads/2023/03/Shot-Predictor-Video.mp4?media=1632743877)
