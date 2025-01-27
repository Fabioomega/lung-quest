# Lung Quest
A simple demo project built on FastAPI to assess general lung health through X-ray images throgh the use of [TorchXrayVision](https://github.com/mlmed/torchxrayvision) and [CXR Lung Risk](https://github.com/AIM-Harvard/CXR-Lung-Risk).

## Quick Start
You need to have the `fastapi[dev]` installed to be able to do this!
If you do not install it like this:
```bash
pip install fastapi[dev]
```

Then run this on the project folder:
```bash
fastapi dev
```

To test if the service is working just run the simple example script:
```bash
python example.py
```

# Disclaimer
I _did not_ implemented a mini-batching system for this api! _I do not recommend to use this demo in production!_
