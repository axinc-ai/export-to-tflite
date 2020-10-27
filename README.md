# tflite-quantization-tutorial

Full integer quantization tutorial using tflite.

## requirements

tensorflow==1.5

## step 1

keras model to tensorflow pb file.

```
python3 keras_to_pb.py
```

## step 2

tensorflow pb to tflite (float).

```
python3 pb_to_tflite.py
```

## step 3

tensorflow pb to tflite (int8).

```
python3 quantize.py
```

## step 4

verificate tflite (float) and tflite (int8).

```
python3 verification.py
```