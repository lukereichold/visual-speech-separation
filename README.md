# About Basis

Basis is a proof-of-concept web app which provides an interactive demonstration of separating on/off-screen audio sources for a given video.

**Live demo available at [basis.fyi](https://basis.fyi)**

It leverages the [speech separation model created by Andrew Owens et al.](http://andrewowens.com/multisensory/) used for separating on / off-screen audio sources. This project is based upon [open-source code and models](https://github.com/andrewowens/multisensory) licensed under the Apache License 2.0.

I built this as an opportunity to learn:

- Implementation details of a "legacy" TensorFlow 1.x model
- How to freeze, inspect, and host a model using TF-Serving
- How to perform real-time inferencing on video from a public web app
