document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("aqi-form");
  const resultSection = document.getElementById("result");
  const aqiValueEl = document.getElementById("aqi-value");
  const aqiCategoryEl = document.getElementById("aqi-category");
  const adviceEl = document.getElementById("advice");
  const predictBtn = document.getElementById("predict-btn");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";

    // collect inputs by name
    const payload = {};
    const inputs = form.querySelectorAll("input");
    inputs.forEach(inp => {
      payload[inp.name] = inp.value;
    });

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      // show result
      resultSection.classList.remove("result-hidden");
      aqiValueEl.textContent = data.aqi;
      aqiCategoryEl.textContent = data.category;
      aqiCategoryEl.style.background = data.color;

      // quick advice
      let advice = "";
      switch (data.category) {
        case "Good":
          advice = "Air quality is good. Normal outdoor activities are fine.";
          break;
        case "Satisfactory":
          advice = "Air is acceptable. Sensitive individuals should take minimal caution.";
          break;
        case "Moderate":
          advice = "Sensitive groups may experience discomfort. Consider limiting prolonged outdoor exertion.";
          break;
        case "Poor":
          advice = "Avoid long outdoor exertion; vulnerable groups should stay indoors.";
          break;
        case "Very Poor":
          advice = "Health warnings of emergency conditions. Stay indoors and avoid outdoor activity.";
          break;
        default:
          advice = "Serious health effects expected. Avoid all outdoor exposure.";
      }
      adviceEl.textContent = advice;
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = "Predict AQI";
    }
  });
});
