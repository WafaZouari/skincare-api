


just follow these steps and your program will run on a localhost
1- pip install -r ./requirements.txt
2- streamlit run main.py  
create venv :python -m venv venv
connect to venv:  ./venv/Scripts/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
this is how to use it later on localhost : "http://127.0.0.1:8000/ask"
use the deployed api from render :"https://skincare-api-iknz.onrender.com"