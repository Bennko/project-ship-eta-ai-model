FROM python:latest
RUN pip install flask
RUN pip install joblib
RUN pip install -U scikit-learn scipy matplotlib
RUN pip install xgboost

COPY ett-service.py ett-service.py
ADD ./models_RotHam ./models_RotHam
ADD ./models_FelRot ./models_FelRot

CMD ["python", "ett-service.py"]