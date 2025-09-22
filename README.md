# PyTorch Docker Application

This project is a Dockerized Python application that utilizes PyTorch for machine learning tasks. Below are the instructions on how to set up and run the application.

## Project Structure

```
pytorch-docker-app
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── src
│   ├── main.py
└── README.md
```

## Prerequisites

- Docker installed on your machine.
- Docker Compose installed.

## Setup Instructions

1. **Clone the Repository**

   Clone this repository to your local machine using:

   ```
   git clone <repository-url>
   ```

2. **Build and Run the Application**

   Use the following command to build and start the application in detached mode:

   ```
   docker-compose -f docker-compose.yml up --build -d
   ```

   This command will build the image if needed and start the application. You should see the output from the `main.py` script in the container logs.

## Dependencies

The project uses the following Python libraries:

- PyTorch
- Any other dependencies specified in `requirements.txt`

Make sure to check `requirements.txt` for the complete list of dependencies.

## Usage

Once the application is running, you can interact with it as specified in the `main.py` file. 




#g++ compiler does not found, so it. was required to add in Dockerfile