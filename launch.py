import os
import argparse
import uvicorn


# Create the parser
parser = argparse.ArgumentParser(description="Launch the ar_sim FastAPI application")

# Add the "data" argument
parser.add_argument("--data", type=str, required=True, help="The data directory")

# Parse the arguments
args = parser.parse_args()

# Set the ARSIM_DATADIR environment variable
os.environ["ARSIM_DATADIR"] = args.data

# Launch the FastAPI application with uvicorn
uvicorn.run("src.ar_sim:ar_sim", host="0.0.0.0", port=8100)