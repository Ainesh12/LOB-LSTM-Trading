
STREAM_KEY=lob:stream
PRED_KEY=lob:predictions
GROUP=lob_group
CONSUMER=c1

MAX_EVENTS=5000
SLEEP_MS=1
FLUSH_EVERY=1000
BUCKETS=10

MODEL_PATH=artifacts/model_state_dict.pt
PARQUET_DIR=data/predictions_parquet

PY=python

.PHONY: help clean redis-clean run producer consumer eval all

help:
	@echo "Targets:"
	@echo "  make redis-clean   # delete redis streams"
	@echo "  make clean         # delete local outputs"
	@echo "  make producer      # replay CSV -> redis stream"
	@echo "  make consumer      # run inference consumer (writes parquet)"
	@echo "  make eval          # evaluate parquet outputs -> eval_summary.csv"
	@echo "  make all           # redis-clean + clean + consumer + producer + eval"

redis-clean:
	redis-cli DEL $(STREAM_KEY) $(PRED_KEY)

clean:
	rm -rf $(PARQUET_DIR)
	rm -f data/predictions_parquet/eval_summary.csv
	mkdir -p $(PARQUET_DIR)

producer:
	$(PY) -m ingestion.replay_csv_to_redis --max_events $(MAX_EVENTS) --sleep_ms $(SLEEP_MS)

consumer:
	$(PY) -m streaming.redis_infer_consumer --model_path $(MODEL_PATH) --flush_every $(FLUSH_EVERY)

eval:
	$(PY) analytics/eval_predictions.py --parquet_dir $(PARQUET_DIR) --buckets $(BUCKETS)

all: redis-clean clean
	@echo "Starting consumer in background..."
	@($(MAKE) consumer &) ; sleep 1
	@echo "Running producer..."
	@$(MAKE) producer
	@echo "Running eval..."
	@$(MAKE) eval
	@echo "Done. See: data/predictions_parquet/eval_summary.csv"
