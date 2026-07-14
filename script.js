/**
 * Spam Mail Classifier - Script
 * 
 * Implements a Multinomial Naive Bayes classifier from scratch in Vanilla JavaScript
 * and trains it using a local CSV dataset.
 */

// Model State
const model = {
    priorSpam: 0.5,
    priorHam: 0.5,
    wordProbSpam: {}, // maps word -> log probability in SPAM
    wordProbHam: {},  // maps word -> log probability in HAM
    vocabulary: new Set(),
    spamWordCount: 0,
    hamWordCount: 0,
    isTrained: false
};

// UI Elements
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const btnClassify = document.getElementById('btnClassify');
const btnClear = document.getElementById('btnClear');
const messageInput = document.getElementById('messageInput');
const validationError = document.getElementById('validationError');
const resultCard = document.getElementById('resultCard');
const resultBadge = document.getElementById('resultBadge');
const resultHeading = document.getElementById('resultHeading');
const resultConfidence = document.getElementById('resultConfidence');

/**
 * 1. Text Cleaning
 * Converts text to lowercase and removes punctuation / special characters.
 */
function cleanText(text) {
    if (!text) return '';
    // Convert text to lowercase
    let cleaned = text.toLowerCase();
    // Replace tabs and newlines with space
    cleaned = cleaned.replace(/[\r\n\t]/g, ' ');
    // Remove unnecessary punctuation (keep letters, numbers, and spaces)
    cleaned = cleaned.replace(/[^\w\s]/g, '');
    return cleaned;
}

/**
 * 2. Tokenize Text
 * Splits cleaned text into individual words (tokens).
 */
function tokenize(text) {
    const cleaned = cleanText(text);
    return cleaned.split(/\s+/).filter(word => word.length > 0);
}

/**
 * 3. CSV Parser (From Scratch)
 * Parses CSV lines, taking into account commas enclosed within double quotes.
 */
function parseCSV(csvText) {
    const lines = [];
    let row = [];
    let field = '';
    let inQuotes = false;

    for (let i = 0; i < csvText.length; i++) {
        const char = csvText[i];

        if (inQuotes) {
            if (char === '"') {
                if (csvText[i + 1] === '"') {
                    // Handled escaped quotes inside quoted string
                    field += '"';
                    i++;
                } else {
                    inQuotes = false;
                }
            } else {
                field += char;
            }
        } else {
            if (char === '"') {
                inQuotes = true;
            } else if (char === ',') {
                row.push(field.trim());
                field = '';
            } else if (char === '\n' || char === '\r') {
                if (char === '\r' && csvText[i + 1] === '\n') {
                    i++;
                }
                row.push(field.trim());
                // Only add non-empty rows
                if (row.length > 0 && row.some(cell => cell !== '')) {
                    lines.push(row);
                }
                row = [];
                field = '';
            } else {
                field += char;
            }
        }
    }

    if (field !== '' || row.length > 0) {
        row.push(field.trim());
        lines.push(row);
    }

    return lines;
}

/**
 * 4. Train Multinomial Naive Bayes Model
 * Computes priors, conditional word frequencies, and applies Laplace smoothing.
 */
function trainNaiveBayes(dataset) {
    if (dataset.length < 2) {
        throw new Error("Dataset is empty or lacks headers.");
    }

    const headers = dataset[0];
    
    // Dynamic Column Identification
    let labelColIndex = 0;
    let textColIndex = 1;

    for (let i = 0; i < headers.length; i++) {
        const header = headers[i].toLowerCase();
        if (header === 'category' || header === 'v1' || header === 'label' || header === 'class' || header === 'target') {
            labelColIndex = i;
        } else if (header === 'message' || header === 'v2' || header === 'text' || header === 'email' || header === 'sms' || header === 'content') {
            textColIndex = i;
        }
    }

    console.log(`Detected columns - Label: "${headers[labelColIndex]}" at index ${labelColIndex}, Text: "${headers[textColIndex]}" at index ${textColIndex}`);

    let totalSpam = 0;
    let totalHam = 0;
    const spamWordFreqs = {};
    const hamWordFreqs = {};

    model.vocabulary.clear();
    model.spamWordCount = 0;
    model.hamWordCount = 0;

    // Process dataset rows (skipping header)
    for (let i = 1; i < dataset.length; i++) {
        const row = dataset[i];
        if (row.length <= Math.max(labelColIndex, textColIndex)) continue;

        const label = row[labelColIndex].toLowerCase().trim();
        const text = row[textColIndex];

        const tokens = tokenize(text);
        if (tokens.length === 0) continue;

        if (label === 'spam') {
            totalSpam++;
            for (const token of tokens) {
                spamWordFreqs[token] = (spamWordFreqs[token] || 0) + 1;
                model.vocabulary.add(token);
                model.spamWordCount++;
            }
        } else {
            totalHam++;
            for (const token of tokens) {
                hamWordFreqs[token] = (hamWordFreqs[token] || 0) + 1;
                model.vocabulary.add(token);
                model.hamWordCount++;
            }
        }
    }

    const totalSamples = totalSpam + totalHam;
    if (totalSamples === 0) {
        throw new Error("No valid training samples found in dataset.");
    }

    // Prior Probabilities
    model.priorSpam = totalSpam / totalSamples;
    model.priorHam = totalHam / totalSamples;

    console.log(`Priors - SPAM: ${model.priorSpam.toFixed(4)}, HAM: ${model.priorHam.toFixed(4)}`);
    console.log(`Vocabulary size: ${model.vocabulary.size}`);
    console.log(`Total words - SPAM: ${model.spamWordCount}, HAM: ${model.hamWordCount}`);

    // Word Probabilities with Laplace Smoothing (+1)
    const vocabSize = model.vocabulary.size;

    model.wordProbSpam = {};
    model.wordProbHam = {};

    for (const word of model.vocabulary) {
        const spamCount = spamWordFreqs[word] || 0;
        const hamCount = hamWordFreqs[word] || 0;

        // Apply Laplace Smoothing and calculate Log Probabilities
        model.wordProbSpam[word] = Math.log((spamCount + 1) / (model.spamWordCount + vocabSize));
        model.wordProbHam[word] = Math.log((hamCount + 1) / (model.hamWordCount + vocabSize));
    }

    model.isTrained = true;
}

/**
 * 5. Predict Message Class
 * Calculates log scores for both classes and decides the winner.
 */
function predictMessage(message) {
    if (!model.isTrained) return null;

    const tokens = tokenize(message);

    // Start with log priors to prevent numerical underflow
    let logScoreSpam = Math.log(model.priorSpam);
    let logScoreHam = Math.log(model.priorHam);

    // Sum log conditional probabilities for words in the vocabulary
    for (const token of tokens) {
        if (model.vocabulary.has(token)) {
            logScoreSpam += model.wordProbSpam[token];
            logScoreHam += model.wordProbHam[token];
        }
    }

    const prediction = logScoreSpam > logScoreHam ? 'spam' : 'ham';

    return {
        prediction,
        logScoreSpam,
        logScoreHam
    };
}

/**
 * 6. Calculate Confidence
 * Safely normalizes Naive Bayes log scores into probabilities.
 */
function calculateConfidence(logScoreSpam, logScoreHam) {
    // Softmax trick: Subtract the maximum log score to avoid overflow/underflow
    const maxScore = Math.max(logScoreSpam, logScoreHam);
    const expSpam = Math.exp(logScoreSpam - maxScore);
    const expHam = Math.exp(logScoreHam - maxScore);
    const sumExp = expSpam + expHam;

    const pSpam = expSpam / sumExp;
    const pHam = expHam / sumExp;

    // Return the probability corresponding to the larger log score
    return logScoreSpam > logScoreHam ? pSpam : pHam;
}

/**
 * 7. Display Prediction Result
 * Renders the classification results on a premium card.
 */
function displayResult(prediction, confidence) {
    resultCard.className = 'result-card'; // Reset classes
    resultCard.classList.add(prediction);

    if (prediction === 'spam') {
        resultBadge.textContent = 'SPAM';
        resultHeading.textContent = 'Likely spam.';
    } else {
        resultBadge.textContent = 'HAM';
        resultHeading.textContent = 'Looks safe.';
    }

    if (confidence !== null && !isNaN(confidence)) {
        const confPercent = Math.round(confidence * 100);
        resultConfidence.textContent = `${confPercent}% confidence`;
    } else {
        resultConfidence.textContent = '';
    }

    resultCard.classList.remove('hidden');
}

/**
 * Load Dataset and Train Model on Page Load
 */
async function loadDataset() {
    try {
        console.log("Loading dataset...");
        // Set UI to training state
        statusDot.className = 'status-dot training';
        statusText.textContent = 'Training model...';
        btnClassify.disabled = true;

        const response = await fetch('./data/dataset.csv');
        if (!response.ok) {
            throw new Error(`Failed to fetch dataset.csv: ${response.statusText}`);
        }

        const csvText = await response.text();
        const parsedData = parseCSV(csvText);

        trainNaiveBayes(parsedData);

        // Update UI to ready state
        statusDot.className = 'status-dot ready';
        statusText.textContent = 'Model Ready';
        btnClassify.disabled = false;
        console.log("Model successfully trained and ready.");
    } catch (error) {
        console.error("Dataset loading/parsing error:", error);
        statusDot.className = 'status-dot';
        statusDot.style.backgroundColor = 'var(--color-error-text)';
        statusText.textContent = 'Error training model. Please check dataset.';
        validationError.textContent = 'Failed to initialize spam classifier model. See console for details.';
    }
}

// Event Listeners
btnClassify.addEventListener('click', () => {
    const text = messageInput.value.trim();

    // Reset error validation
    validationError.textContent = '';

    if (text === '') {
        validationError.textContent = 'Please enter an email or message.';
        resultCard.classList.add('hidden');
        return;
    }

    const result = predictMessage(text);
    if (result) {
        const confidence = calculateConfidence(result.logScoreSpam, result.logScoreHam);
        displayResult(result.prediction, confidence);
    }
});

btnClear.addEventListener('click', () => {
    messageInput.value = '';
    validationError.textContent = '';
    resultCard.classList.add('hidden');
});

// Initial Load
document.addEventListener('DOMContentLoaded', loadDataset);
