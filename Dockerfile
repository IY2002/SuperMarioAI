FROM python:3.8

ADD main.py .

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install stable-baselines3[extra] gym_super_mario_bros==7.3.0

CMD ["python", "./main.py"]