# Spam Mail Classifier using Multinomial Naive Bayes

A simple, interactive, web-based Spam Mail Classifier that classifies user-entered text (emails, messages, SMS) as **SPAM** or **HAM** (safe). This project is designed to be beginner-friendly, premium in aesthetics, and suitable for a college Machine Learning mini-project. 

The Multinomial Naive Bayes classifier is implemented using Vanilla JavaScript and trained directly in the browser using the provided spam/ham dataset.

---

## Live Preview & Visuals
- **Frontend-only**: Fully client-side, zero backend APIs or dependencies.
- **Apple-inspired Design**: Clean spacing, large typography, minimalist neutral colors, and responsive layouts.
- **Deployable**: Ready for hosting on **GitHub Pages**.

---

## Technologies Used
- **HTML5**: For semantic layout.
- **Vanilla CSS**: Styled with CSS variables, responsive typography using `clamp()`, and smooth flex/grid properties.
- **Vanilla JavaScript (ES6)**: Implement CSV parser, text preprocessing, and Multinomial Naive Bayes algorithm from scratch.

---

## Machine Learning Concepts

### 1. What is Multinomial Naive Bayes?
Naive Bayes is a family of probabilistic classifiers based on **Bayes' Theorem** with the "naive" assumption of conditional independence between every pair of features (words) given the class label. 

The **Multinomial Naive Bayes** variant is particularly suited for text classification, where features represent word frequencies (counts) in a document.

### 2. Bayes' Theorem Formula
To find the probability that a message belongs to a class $C$ (where $C \in \{\text{SPAM}, \text{HAM}\}$):

$$P(C | \text{Message}) = \frac{P(\text{Message} | C) \cdot P(C)}{P(\text{Message})}$$

Since $P(\text{Message})$ is constant for all classes, we classify by finding the class $C$ that maximizes:

$$\hat{C} = \arg\max_{C} P(\text{Message} | C) \cdot P(C)$$

For a message containing words $(w_1, w_2, \dots, w_n)$, the conditional independence assumption lets us write:

$$P(\text{Message} | C) = P(w_1 | C) \cdot P(w_2 | C) \dots P(w_n | C) = \prod_{i=1}^{n} P(w_i | C)$$

### 3. Laplace Smoothing (+1)
If a word in the user's message was never seen in the training data for a class, $P(w_i | C) = 0$, which would make the entire product zero. To solve this, we apply **Laplace Smoothing** by adding a value of $\alpha = 1$ to the counts:

$$P(w | C) = \frac{\text{count}(w | C) + 1}{N_C + |V|}$$

Where:
- $\text{count}(w | C)$ is the frequency of word $w$ in class $C$ messages.
- $N_C$ is the total count of all words in class $C$.
- $|V|$ is the size of the vocabulary (total number of unique words in the training set).

### 4. Preventing Numerical Underflow (Log Probabilities)
Multiplying many tiny probabilities (e.g., $P(w_i | C) \approx 0.0001$) results in numbers so close to zero that computers lose precision (numerical underflow). To prevent this, we calculate the sum of **log-probabilities** instead of the product of raw probabilities:

$$\ln(P(C | \text{Message})) \propto \ln(P(C)) + \sum_{i=1}^{n} \ln(P(w_i | C))$$

Since $\ln(x)$ is a monotonically increasing function, the class with the highest log score is still the class with the highest probability.

### 5. Confidence Score Calculation
To display a user-friendly percentage confidence, we normalize the log scores back into probability space using the **Softmax normalization** technique:

$$P(C_k | \text{Message}) = \frac{e^{\text{Score}(C_k)}}{\sum_{j} e^{\text{Score}(C_j)}}$$

To prevent overflow when computing exponentials, we subtract the maximum log score before exponentiating:

$$\text{Score}'(C) = \text{Score}(C) - \max(\text{Scores})$$
$$P(C | \text{Message}) = \frac{e^{\text{Score}'(C)}}{e^{\text{Score}'(\text{SPAM})} + e^{\text{Score}'(\text{HAM})}}$$

---

## Project Structure
```
spam-mail-classifier/
│
├── index.html       # Webpage structure & UI components
├── style.css        # Apple-inspired premium minimalist design styles
├── script.js        # CSV parser, text cleaner, and Naive Bayes ML implementation
├── data/
│   └── dataset.csv  # The CSV file containing the Spam and Ham SMS data
└── README.md        # Technical explanation and setup guide (this file)
```

---

## How the Model Trains in the Browser
When the page loads:
1. `loadDataset()` triggers an asynchronous `fetch('./data/dataset.csv')` request.
2. The UI switches to `Training model...` status and disables the Classify button.
3. Once fetched, `parseCSV()` converts the raw CSV string into a JavaScript array (correctly parsing commas inside quotes).
4. `trainNaiveBayes()` parses the array:
   - Identifies text and target columns dynamically.
   - Cleans the text (lowercases, removes punctuation).
   - Counts prior occurrences of SPAM and HAM.
   - Generates the vocabulary and maps word occurrences to each class.
   - Saves smoothed log-probabilities for all vocabulary words.
5. The UI updates to `Model Ready` and the user is allowed to enter messages for classification.

---

## How to Run Locally

### 1. Clone the repository or download the files
Place all files inside a single directory matching the Project Structure.

### 2. Run a local server
Because of browser security policies (`CORS`), fetching local files (`./data/dataset.csv`) directly via `file://` protocol will result in an error. You must serve the directory using a simple local server.

- **Option A (Python)**:
  Open a terminal inside your folder and run:
  ```bash
  python -m http.server 8000
  ```
  Then open `http://localhost:8000` in your web browser.

- **Option B (VS Code Live Server Extension)**:
  If using VS Code, click the **"Go Live"** button in the status bar to launch a server automatically.

---

## How to Deploy on GitHub Pages
1. Push your project folder to a new public repository on GitHub.
2. Go to the repository **Settings** page.
3. Click on the **Pages** tab on the left sidebar (under "Code and automation").
4. Under **Build and deployment**, select **Deploy from a branch**.
5. Select the `main` or `master` branch and folder `/ (root)`.
6. Click **Save**.
7. In a minute, GitHub will provide a live URL (e.g., `https://username.github.io/repository-name/`) where your classifier is fully accessible.
