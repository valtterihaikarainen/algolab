### Specification Document 

---

### Administrative information

- **Study programme**: Bachelor’s in Mathematics (Theoretical Computer Science)
- **Project documentation language**: English.  

---

### Programming languages

- **Primary implementation language**: **C++** (C++17 or later).  
- I review projects written in **Python** and **C++** 

---

### Algorithms and data structures to be implemented

- **Feed‑forward neural network (MLP)** for MNIST classification:
  - Input layer of size 784 (flattened 28×28 image).
  - One or two hidden layers with nonlinearity (e.g. ReLU or tanh).
  - Output layer with 10 units and softmax.

- **From‑scratch backpropagation**:
  - Forward and backward pass for:
    - Dense linear layer (matrix–vector product + bias).
    - Butterfly linear layer (see below).
    - Activation functions (ReLU/tanh) and softmax + cross‑entropy loss.
  - Parameter updates using (stochastic) gradient descent or simple SGD with momentum.

- **Butterfly‑structured linear layer**:
  - A linear layer where the weight matrix is **factorized into butterfly stages**:
    - Each stage is block‑diagonal with small \(b \times b\) blocks (e.g. \(b \in \{2, 4, 16\}\)) and a fixed “butterfly” connectivity pattern between stages.
    - This generalizes the FFT butterfly structure but with learnable block parameters.
  - Full forward and backward implementations in C++, including gradient computation w.r.t. block parameters and input activations.

- **Data handling and simple containers**:
  - Storage of parameters and activations using either:
    - Standard library containers (`std::vector`) and manual indexind
  - Mini‑batch loading and normalization for MNIST.

No automatic differentiation or high‑level deep learning libraries (e.g. PyTorch, TensorFlow) will be used; all gradient calculations will be coded manually.

---

### Problem being solved

The project studies whether **butterfly‑structured linear layers** can be used as a drop‑in replacement for standard dense layers in a small neural network trained on MNIST, when everything is implemented from scratch in C++. Dense layers are simple but use \(O(n^2)\) parameters and do not exploit possible structure in the weight matrix. Butterfly layers factor the weight matrix into a product of sparse stages with small blocks, potentially reducing parameter count and operation count.

The core question is: **Can a small MLP with butterfly layers, trained via manual backpropagation in C++, achieve reasonable classification accuracy on MNIST, and how does it compare to a standard dense MLP in terms of accuracy and complexity?**

---

### Inputs and how they are used

- **MNIST dataset**:
  - Training and test images (28×28 grayscale) and labels (0–9).
  - Images are flattened into 784‑dimensional vectors and optionally normalized (e.g. pixel values scaled to \([0,1]\)).
  - Used as input–label pairs for training and evaluating the network.

- **Configuration parameters**:
  - Network architecture (hidden layer size, choice of activation).
  - Whether the first (and/or second) linear layer is **dense** or **butterfly‑structured**.
  - Butterfly hyperparameters:
    - Block size \(b\) (e.g. 2, 4, 16).
    - Number of butterfly stages \(L\).
  - Training hyperparameters:
    - Learning rate, batch size, number of epochs, optional momentum.

During training, the program repeatedly:

1. Reads mini‑batches of MNIST images and labels.  
2. Runs the forward pass through the network (including butterfly or dense layers).  
3. Computes softmax cross‑entropy loss.  
4. Runs backpropagation to compute gradients.  
5. Updates parameters using SGD.

At the end, it evaluates accuracy on the MNIST test set and prints summary metrics.

---

### Expected time and space complexities

Let:

- \(n\) be the input or hidden layer width (e.g. 784 or 128),  
- \(m\) be the next layer width,  
- \(b\) be the butterfly block size,  
- \(L\) be the number of butterfly stages,  
- \(B\) be the mini‑batch size, and  
- \(T\) be the total number of training steps (batches over all epochs).

**Dense linear layer**

- Forward (per batch):
  - Time: \(\Theta(B \cdot n \cdot m)\) operations (matrix–batch multiplication).
- Backward (per batch):
  - Time: \(\Theta(B \cdot n \cdot m)\) to compute gradients w.r.t. weights and inputs.
- Parameters:
  - \(\Theta(n \cdot m)\) weights + \(\Theta(m)\) biases.

**Butterfly linear layer**

Assume an \(n \times n\) butterfly layer with \(L\) stages and block size \(b\) (for simplicity, applied to a hidden layer of width \(n\)).

- Each stage:
  - Has \(n/b\) independent \(b \times b\) blocks.
  - Forward cost per vector: \(\Theta((n/b) \cdot b^2) = \Theta(n \cdot b)\).
- Forward (per batch):
  - Time: \(\Theta(B \cdot L \cdot n \cdot b)\).
- Backward (per batch):
  - Similar order: \(\Theta(B \cdot L \cdot n \cdot b)\) to propagate gradients through all stages and blocks.
- Parameters:
  - Per stage: \(\Theta((n/b) \cdot b^2) = \Theta(n \cdot b)\).
  - Total: \(\Theta(L \cdot n \cdot b)\) parameters (plus \(\Theta(n)\) biases if used).

For moderate block sizes \(b\) and \(L = O(\log n)\), butterfly layers can have fewer parameters and potentially lower per‑vector cost than dense layers of comparable width.

**Backpropagation and training loop**

- Per training step (one mini‑batch), time is dominated by the forward and backward passes of all layers:
  - For a simple 2‑layer MLP, roughly:
    - Dense baseline: \(\Theta(B \cdot n \cdot m)\) per layer, twice for forward + backward.
    - Butterfly variant: replace one dense layer by \(\Theta(B \cdot L \cdot n \cdot b)\).
- Total training time:
  - \(\Theta(T \cdot \text{cost\_per\_step})\), where \(T\) is the total number of mini‑batches.
- Space:
  - Parameters and intermediate activations:
    - \(\Theta(n \cdot m)\) for dense layers,
    - \(\Theta(L \cdot n \cdot b)\) for butterfly layers,
    - plus \(\Theta(B \cdot n)\) for activations stored during backprop.

I will use typical MNIST sizes (e.g. \(n = 784\), hidden size 128, batch sizes like 32 or 64, moderate \(L\) and \(b\)), which should keep training time and memory usage well within the limits for this course.

---

### List of sources I intend to use

Preliminary list (to be expanded and properly cited in the final report):

- Standard material on feed‑forward neural networks and backpropagation:
  - Michael Nielsen, *“Neural Networks and Deep Learning”* (online book).
  - Selected sections from Goodfellow et al., *“Deep Learning”*.
- Resources on butterfly and structured linear layers:
  - Tri Dao et al., **“Learning Fast Algorithms for Linear Transforms”**, ICML 2019.  
  - Tri Dao et al., **“Monarch Matrices: Expressive Structured Matrices from Simple Building Blocks”**, ICML 2022.  
  - Classical FFT / butterfly network explanations (textbook or online).
- MNIST dataset documentation and description.
- C++ references for efficient numerical programming and (if used) Eigen/Armadillo documentation.

---

### Core of my project (short description)

The core of my project is to **implement from scratch in C++ a small feed‑forward neural network with both dense and butterfly‑structured linear layers, including full backpropagation and training on MNIST**, and to compare their behavior.

I will first build a baseline MLP with standard dense layers and manual backprop, and train it to classify MNIST digits. Then I will implement a butterfly linear layer, where the weight matrix is expressed as a product of sparse stages with small \(b \times b\) blocks following a butterfly pattern, and derive and code its forward and backward passes. By replacing one of the dense layers with a butterfly layer and training this network on MNIST, I will measure classification accuracy, parameter counts, and per‑forward‑pass operation counts. This allows me to study, in a controlled and fully self‑implemented setting, how a structured butterfly layer compares to a standard dense layer in a simple but realistic machine‑learning task.
