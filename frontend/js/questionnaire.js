const form = document.querySelector("#questionnaire-form");
const fieldsContainer = document.querySelector("#question-fields");
const stepCounter = document.querySelector("#step-count");
const stepName = document.querySelector("#step-name");
const progressBar = document.querySelector("#progress-fill");
const progressTrack = document.querySelector(".progress-track");
const previousButton = document.querySelector("#previous-step");
const nextButton = document.querySelector("#next-step");
const submitButton = document.querySelector("#submit-assessment");
const errorBox = document.querySelector("#question-error");
const loadingState = document.querySelector("#loading-state");

const groups = [
  {
    id: "background",
    title: "Basic information",
    description: "Enter the age field represented in the source dataset.",
  },
  {
    id: "lifestyle",
    title: "Lifestyle",
    description: "Smoking is an input in the saved model. Choose Don't know if you are unsure.",
  },
  {
    id: "medical-history",
    title: "Medical history",
    description: "This sensitive history field is included because it is an input in the model. Select Don't know if you are unsure.",
  },
];

let questionnaire = [];
let displayedQuestions = [];
let currentStep = 0;

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.textContent = "";
  errorBox.hidden = true;
}

function buildNumberField(feature, index) {
  const wrapper = document.createElement("fieldset");
  wrapper.className = "question-field";
  const legend = document.createElement("legend");
  legend.append(document.createTextNode(feature.question || feature.label));
  const required = document.createElement("span");
  required.className = "required-label";
  required.textContent = "Required";
  legend.append(required);
  wrapper.append(legend);

  const input = document.createElement("input");
  input.type = "number";
  input.id = `question-${index}`;
  input.name = feature.name;
  input.min = String(feature.min);
  input.max = String(feature.max);
  input.step = feature.step === 1 ? "1" : "any";
  input.required = feature.allow_unknown === false;
  input.autocomplete = "off";
  input.setAttribute("aria-label", feature.question || feature.label);
  input.setAttribute("aria-describedby", `range-${index}`);
  wrapper.append(input);

  const range = document.createElement("p");
  range.className = "field-help";
  range.id = `range-${index}`;
  range.textContent = `Dataset range: ${feature.min} to ${feature.max}`;
  wrapper.append(range);

  if (feature.allow_unknown !== false) {
    const unknownLabel = document.createElement("label");
    unknownLabel.className = "unknown-option";
    const unknown = document.createElement("input");
    unknown.type = "checkbox";
    unknown.name = `unknown-${index}`;
    unknown.addEventListener("change", () => {
      input.disabled = unknown.checked;
      input.required = !unknown.checked;
      if (unknown.checked) input.value = "";
      clearError();
    });
    unknownLabel.append(unknown, document.createTextNode("Not sure"));
    wrapper.append(unknownLabel);
  }
  return wrapper;
}

function buildBooleanField(feature, index) {
  const wrapper = document.createElement("fieldset");
  wrapper.className = "question-field";
  const legend = document.createElement("legend");
  legend.append(document.createTextNode(feature.question || feature.label));
  const required = document.createElement("span");
  required.className = "required-label";
  required.textContent = "Required";
  legend.append(required);
  wrapper.append(legend);
  if (feature.help) {
    const help = document.createElement("p");
    help.className = "field-help";
    help.textContent = feature.help;
    wrapper.append(help);
  }

  const choices = document.createElement("div");
  choices.className = "choice-row";
  choices.setAttribute("role", "radiogroup");
  choices.setAttribute("aria-label", feature.question || feature.label);
  const options = [
    { label: "Yes", value: "1" },
    { label: "No", value: "0" },
  ];
  if (feature.allow_unknown) {
    options.push({ label: "Don't know", value: "" });
  }
  options.forEach((option, optionIndex) => {
    const id = `question-${index}-${optionIndex}`;
    const choice = document.createElement("span");
    choice.className = "choice-option";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = feature.name;
    input.id = id;
    input.value = option.value;
    input.required = optionIndex === 0;
    input.addEventListener("change", clearError);
    const label = document.createElement("label");
    label.htmlFor = id;
    label.textContent = option.label;
    choice.append(input, label);
    choices.append(choice);
  });
  wrapper.append(choices);
  return wrapper;
}

function renderQuestionnaire() {
  groups.forEach((group, stepIndex) => {
    const section = document.createElement("section");
    section.className = "question-section";
    section.dataset.step = String(stepIndex);
    section.hidden = stepIndex !== currentStep;
    section.setAttribute("aria-labelledby", `group-title-${stepIndex}`);

    const heading = document.createElement("h2");
    heading.id = `group-title-${stepIndex}`;
    heading.textContent = group.title;
    const description = document.createElement("p");
    description.textContent = group.description;
    const grid = document.createElement("div");
    grid.className = "question-grid";

    displayedQuestions
      .filter((feature) => feature.group === group.id)
      .forEach((feature) => {
        const index = displayedQuestions.indexOf(feature);
        grid.append(
          feature.type === "boolean"
            ? buildBooleanField(feature, index)
            : buildNumberField(feature, index),
        );
      });

    section.append(heading, description, grid);
    fieldsContainer.append(section);
  });
  updateStep();
}

function validateCurrentStep() {
  const section = fieldsContainer.querySelector(`[data-step="${currentStep}"]`);
  for (const field of section.querySelectorAll(".question-field")) {
    const radios = field.querySelectorAll('input[type="radio"]');
    if (radios.length && !Array.from(radios).some((radio) => radio.checked)) {
      radios[0].reportValidity();
      return false;
    }
    const numberInput = field.querySelector('input[type="number"]');
    if (numberInput && !numberInput.disabled && !numberInput.checkValidity()) {
      numberInput.reportValidity();
      return false;
    }
  }
  return true;
}

function updateStep() {
  const sections = fieldsContainer.querySelectorAll(".question-section");
  sections.forEach((section, index) => {
    section.hidden = index !== currentStep;
  });
  stepCounter.textContent = `STEP ${currentStep + 1} OF ${groups.length}`;
  stepName.textContent = groups[currentStep].title;
  progressBar.style.width = `${((currentStep + 1) / groups.length) * 100}%`;
  progressTrack.setAttribute("aria-valuemax", String(groups.length));
  progressTrack.setAttribute("aria-valuenow", String(currentStep + 1));
  previousButton.disabled = currentStep === 0;
  nextButton.hidden = currentStep === groups.length - 1;
  submitButton.hidden = currentStep !== groups.length - 1;
  clearError();
}

function collectResponses() {
  const responses = {};
  questionnaire.forEach((feature) => {
    if (feature.type === "boolean") {
      const selected = form.querySelector(`input[name="${CSS.escape(feature.name)}"]:checked`);
      responses[feature.name] = selected && selected.value !== "" ? Number(selected.value) : null;
      return;
    }
    const input = form.querySelector(`input[name="${CSS.escape(feature.name)}"]`);
    responses[feature.name] = input.disabled || input.value === "" ? null : input.valueAsNumber;
  });
  return responses;
}

previousButton.addEventListener("click", () => {
  if (currentStep > 0) {
    currentStep -= 1;
    updateStep();
  }
});

nextButton.addEventListener("click", () => {
  if (!validateCurrentStep()) return;
  currentStep += 1;
  updateStep();
  document.querySelector(".question-card").scrollIntoView({ behavior: "smooth", block: "start" });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!validateCurrentStep()) return;
  clearError();
  submitButton.disabled = true;
  previousButton.disabled = true;
  loadingState.hidden = false;

  try {
    const responses = collectResponses();
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ responses }),
    });
    const result = await response.json();
    if (!response.ok || !result.success) {
      throw new Error(result.error || "The assessment could not be completed.");
    }
    try {
      sessionStorage.setItem(
        "cervicare-result",
        JSON.stringify({ result, responses }),
      );
    } catch {
      throw new Error("Your browser blocked temporary session storage, so this tab cannot display the result.");
    }
    window.location.assign("/result");
  } catch (error) {
    showError(
      error instanceof TypeError
        ? "Could not reach CerviCare. Check that the Flask application is running, then try again."
        : error.message,
    );
    submitButton.disabled = false;
    previousButton.disabled = currentStep === 0;
    loadingState.hidden = true;
  }
});

async function loadQuestionnaire() {
  try {
    const response = await fetch("/api/features");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Questionnaire fields could not be loaded.");
    questionnaire = data.questionnaire;
    if (!Array.isArray(questionnaire) || questionnaire.length === 0) {
      throw new Error("Questionnaire fields are unavailable.");
    }
    displayedQuestions = questionnaire;
    renderQuestionnaire();
  } catch (error) {
    showError(
      error instanceof TypeError
        ? "Could not reach CerviCare. Check that the Flask application is running."
        : error.message,
    );
    nextButton.disabled = true;
  }
}

loadQuestionnaire();
