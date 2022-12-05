# tflite-quantization-tutorial

Full integer quantization tutorial using tflite.

# from TensorFlow Keras

## requirements

tensorflow==1.5 or tensorflow==2.6

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

# From torch

## requirements

- torch
- tensorflow2
- onnx2tf https://github.com/PINTO0309/onnx2tf

## step 1

torch model to onnx.

```
python3 torch_to_onnx.py
```

## step 2

onnx to pb and tflite (float).

```
python3 -m onnx2tf -i efficientnetlite.onnx -osd
```

## step 3

pb to tflite (int8).

```
python3 quantize.py
```

## step 4

verificate tflite (float) and tflite (int8).

```
python3 verification.py
```