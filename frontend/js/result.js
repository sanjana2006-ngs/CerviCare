const resultCard = document.querySelector("#result-card");
const classificationText = document.querySelector("#classification");
const guidanceText = document.querySelector("#result-guidance");
const resultError = document.querySelector("#result-error");

try {
  const savedAssessment = sessionStorage.getItem("cervicare-result");
  if (!savedAssessment) {
    throw new Error("No assessment result is available in this browser session. Start a new assessment to continue.");
  }

  const assessment = JSON.parse(savedAssessment);
  const { result, responses } = assessment;
  if (
    !result.success
    || ![0, 1].includes(result.prediction)
    || !responses
    || result.classification !== `Model predicts ${result.prediction === 1 ? "positive" : "negative"} biopsy class.`
  ) {
    throw new Error("The saved result could not be displayed. Please start a new assessment.");
  }

  classificationText.textContent = result.classification;
  document.querySelector("#report-classification").textContent = result.classification;
  const isPositive = result.prediction === 1;
  if (isPositive) {
    resultCard.classList.add("is-positive");
    guidanceText.textContent = "This result is not a diagnosis. Please discuss screening concerns with a qualified healthcare professional.";
  } else {
    guidanceText.textContent = "A negative prediction does not rule out disease. Follow recommended screening guidance and discuss concerns with a healthcare professional.";
  }

  const labels = {
    Age: "Age",
    Smokes: "Smoking",
    STDs: "STI history",
  };
  const formatValue = (name, value) => {
    if (value === null || value === undefined) return "Don't know";
    if (name === "Age") return `${value} years`;
    return value === 1 ? "Yes" : "No";
  };
  const inputList = document.querySelector("#result-inputs");
  const reportInputList = document.querySelector("#report-inputs");
  Object.entries(labels).forEach(([name, label]) => {
    const item = document.createElement("li");
    item.textContent = `${label}: ${formatValue(name, responses[name])}`;
    inputList.append(item);
    reportInputList.append(item.cloneNode(true));
  });
  document.querySelector("#report-date").textContent = new Intl.DateTimeFormat(
    undefined,
    { dateStyle: "long" },
  ).format(new Date());
} catch (error) {
  resultError.textContent = error.message;
  resultError.hidden = false;
  classificationText.textContent = "Result unavailable";
  guidanceText.textContent = "Start a new assessment to generate a result from the saved model.";
}

document.querySelector("#print-report-button").addEventListener("click", () => {
  window.print();
});
