FROM rayproject/ray:02a343-py38-aarch64

ENV TASK_ROOT=/var/task
ENV PATH="$TASK_ROOT/bin:$PATH"

COPY requirements.txt ${TASK_ROOT}/fastapi/requirements.txt

RUN pip3 install --no-cache-dir -r ${TASK_ROOT}/fastapi/requirements.txt --break-system-packages

WORKDIR ${TASK_ROOT}
    
EXPOSE 8888

COPY main.py ${TASK_ROOT}/fastapi/main.py
COPY serve.py ${TASK_ROOT}/fastapi/serve.py

CMD ["bash", "-c", "PYTHONPATH=${TASK_ROOT}/fastapi uvicorn main:app --host 0.0.0.0 --port 8888 --log-level debug --timeout-keep-alive 300"]

