# auto_ml

The app is developed to show the capabilities of pycaret library
for Machine Learning pipelines. Web interface is designed with Streamlit framework.

The project is dockerized. So, to launch it you should do the following:

1) Pull the files to your local repository;
2) Make sure your current working directory is the project root folder;
3) Make sure Docker is installed on your computer;
4) run ```docker build -f Dockerfile -t app:latest .``` command in terminal to build a docker image;
5) run ```docker run -p 8501:8501 app:latest``` command in terminal to start docker container;
6) navigate to ```http://localhost:8501/``` in your browser.

To start modeling you need to upload data.gzip file from project root folder using web interface 
(there is no need to unpack the archive).

Note: the app is not designed for external data files. Use .gzip file only provided in project folder 
unless you are sure that your own data file meets the requirements of preprocessing pipeline.

Enjoy!
