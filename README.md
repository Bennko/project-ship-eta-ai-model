## Setup (currently not possible due to missing binary files)

To set up the system naviagate to the folder product and with docker compose installed type in following command:

sudo docker compose up

Then, when both of the servers have deployed successfully then you can access the application in the browse under:

http://localhost:3000/

Our folder structure and what they contain:
architecture: documentation purposes
data: raw and cleaned data for both routes
machine_learning: 
    for each model we considered through our process, we have a folder with test files. In those we played around with the hyperparameters so find the best models suited for our usecase
    Stacking: here we implemented a combination of the three base models via a meta model and the models saved in files via joblib. Also, for every route there is a light version (smaller models) that we used in our product to achieved faster runtimes
product: server infrastructure for the frontend and the docker containers
