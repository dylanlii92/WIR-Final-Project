const searchForm = document.getElementById("search-form");
const searchInput = document.getElementById("search-input");
const resultsContainer = document.getElementById("results");
const statusContainer = document.getElementById("status");

searchForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const query = searchInput.value.trim();

    if (!query) {
        return;
    }

    statusContainer.textContent = "Searching for music events...";
    resultsContainer.innerHTML = "";

    try {
        const response = await fetch(`/ask?q=${encodeURIComponent(query)}`);

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Search failed");
        }

        displayEvents(data.results);

    } catch (error) {
        console.error(error);

        statusContainer.textContent =
            "Could not fetch events.";
    }
});

function displayEvents(events) {
    resultsContainer.innerHTML = "";

    if (events.length === 0) {
        statusContainer.textContent = "No events found.";
        return;
    }

    statusContainer.textContent =
        `Found ${events.length} events`;

    events.forEach(function (event) {

        const card = document.createElement("article");

        card.className = "event-card";

        card.innerHTML = `
            <img
                src="${event.image}"
                alt="${event.name}"
            >

            <div class="event-content">
                <h2>${event.name}</h2>

                <p>
                    <strong>Date:</strong>
                    ${event.date}
                </p>

                <p>
                    <strong>Venue:</strong>
                    ${event.venue}
                </p>

                <p>
                    <strong>City:</strong>
                    ${event.city}
                </p>

                <p>
                    ${event.explanation}
                </p>

                <a
                    href="${event.url}"
                    target="_blank"
                >
                    View tickets
                </a>
            </div>
        `;

        resultsContainer.appendChild(card);
    });
}