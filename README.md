# What is this?

A demo RNN written with Tensorflow 1.0.0
Dummy data is used as timeseries, and the model tries to predict the last portion
of the given series.

# Run

- Edit `fixtures/config.yaml`
- `docker-compose up train`
- check `./outputs/`

# Tests

- `docker-compose up test`

