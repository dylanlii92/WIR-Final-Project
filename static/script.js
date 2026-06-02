const searchForm = document.getElementById("search-form");
const searchInput = document.getElementById("search-input");
const resultsContainer = document.getElementById("results");
const statusContainer = document.getElementById("status");
const showMoreButton = document.getElementById("show-more");

const RESULTS_PER_PAGE = 6;
let currentQuery = "";
let nextOffset = 0;
let visibleEventCount = 0;
let isLoadingMore = false;

searchForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const query = searchInput.value.trim();

    if (!query) {
        return;
    }

    statusContainer.textContent = "Searching for music events...";
    resultsContainer.innerHTML = "";
    showMoreButton.hidden = true;
    currentQuery = query;
    nextOffset = 0;
    visibleEventCount = 0;

    try {
        await fetchEventsPage({ append: false });
    } catch (error) {
        console.error(error);
        statusContainer.textContent = "Could not fetch events.";
    }
});

searchInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        searchForm.requestSubmit();
    }
});

showMoreButton.addEventListener("click", async function () {
    if (isLoadingMore || !currentQuery) {
        return;
    }

    isLoadingMore = true;
    showMoreButton.disabled = true;
    showMoreButton.textContent = "Loading...";

    try {
        await fetchEventsPage({ append: true });
    } catch (error) {
        console.error(error);
        statusContainer.textContent = "Could not fetch more events.";
    } finally {
        isLoadingMore = false;
        showMoreButton.disabled = false;
        showMoreButton.textContent = "Show more";
    }
});

async function fetchEventsPage({ append }) {
    const params = new URLSearchParams({
        q: currentQuery,
        limit: RESULTS_PER_PAGE,
        offset: nextOffset
    });
    const response = await fetch(`/ask?${params.toString()}`);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || "Search failed");
    }

    const events = data.results || [];

    if (!append) {
        resultsContainer.innerHTML = "";
    }

    if (!append && events.length === 0) {
        statusContainer.textContent = "No events found.";
        showMoreButton.hidden = true;
        return;
    }

    if (append && events.length === 0) {
        showMoreButton.hidden = true;
        return;
    }

    appendEvents(events);

    visibleEventCount += events.length;
    nextOffset = data.next_offset || visibleEventCount;
    statusContainer.textContent = `Showing ${visibleEventCount} events`;
    showMoreButton.hidden = !data.has_more && events.length < RESULTS_PER_PAGE;
}

function appendEvents(events) {
    events.forEach(function (event) {
        resultsContainer.appendChild(createEventCard(event));
    });
}

function createEventCard(event) {
    const card = document.createElement("article");
    card.className = "event-card";
    const whyText = getRecommendationReason(event);
    const matchLabels = getMatchLabels(event);

    card.innerHTML = `
        <img
            src="${escapeHtml(event.image || "https://via.placeholder.com/400x250?text=Music+Event")}"
            alt="${escapeHtml(event.name || "Music event")}"
        >

        <div class="event-content">
            <h2>${escapeHtml(event.name || "Unknown event")}</h2>

            <p><strong>Date:</strong> ${escapeHtml(event.date || "Unknown")}</p>

            <p><strong>Venue:</strong> ${escapeHtml(event.venue || "Unknown")}</p>

            <p><strong>City:</strong> ${escapeHtml(event.city || "Unknown")}</p>

            <p><strong>Genre:</strong> ${escapeHtml(event.genre || "Unknown")}</p>

            <p><strong>Subgenre:</strong> ${escapeHtml(event.subgenre || "Unknown")}</p>

            <a href="${escapeHtml(event.url || "#")}" target="_blank">
                View tickets
            </a>

            <details class="ai-explanation">
                <summary>Why this?</summary>

                <div class="ai-explanation-body">
                    <p>${escapeHtml(whyText)}</p>

                    ${matchLabels.length ? `
                        <div class="match-tags">
                            ${matchLabels.map(function (label) {
                                return `<span>${escapeHtml(label)}</span>`;
                            }).join("")}
                        </div>
                    ` : ""}
                </div>
            </details>
        </div>
    `;

    return card;
}

function getRecommendationReason(event) {
    if (shouldUseAiReason(event.llm_reason)) {
        return event.llm_reason;
    }

    const artist = event.artist || "the artist";
    const city = event.city || "this city";
    const venue = event.venue || "the venue";
    const genre = event.genre || "live music";
    const variant = getStableVariant(event, 5);

    if (event.artist_query_match && event.city_query_match) {
        const templates = [
            `${artist} is the strongest match for your search, and this date is specifically in ${city} at ${venue}.`,
            `This one checks the two biggest boxes: the artist you asked for and the city you typed in.`,
            `Because it combines ${artist} with a ${city} stop, it rises above similar shows in other cities.`,
            `The recommendation is centered on exact fit: ${artist}, ${city}, and a real event listing at ${venue}.`,
            `This is likely what you meant: the requested artist playing in the requested city.`
        ];

        return templates[variant];
    }

    if (event.artist_query_match) {
        const templates = [
            `${artist} matches the artist in your search, but this result is outside the requested city.`,
            `This is included mainly because it is the right artist, even though the location is different.`,
            `The artist match is strong here, so it can still be useful if you are open to nearby or alternate cities.`,
            `${artist} is the key reason this appears; the city match is weaker for this card.`,
            `This ranks because the performer is highly relevant to your query.`
        ];

        return templates[variant];
    }

    if (event.city_query_match) {
        const templates = [
            `This event is in ${city}, so it is kept as a local option even though the artist match is not exact.`,
            `The location matches your search, which makes it a useful nearby alternative.`,
            `This is a ${city} recommendation with a ${genre} profile that may fit the vibe of the query.`,
            `It appears because the city is right, giving you another option in the area.`,
            `This card is more about location fit than exact artist fit.`
        ];

        return templates[variant];
    }

    const fallbackTemplates = [
        `This ranked well against the words in your search and the event details available.`,
        `This is a broader suggestion based on genre, title, artist, and description similarity.`,
        `The system included this as a related event that may still fit the intent of your search.`,
        `This result is more exploratory, based on overall similarity rather than one exact match.`,
        `This is a secondary recommendation that may be worth checking if you want more options.`
    ];

    return fallbackTemplates[variant];
}

function getMatchLabels(event) {
    const labels = [];

    if (event.artist_query_match) {
        labels.push("Artist match");
    }

    if (event.city_query_match) {
        labels.push("City match");
    }

    if (event.criteria_count) {
        labels.push(`${event.criteria_count} criteria matched`);
    }

    if (event.explanation) {
        labels.push(event.explanation);
    }

    return labels;
}

function shouldUseAiReason(reason) {
    if (!reason || reason === "No reason provided") {
        return false;
    }

    const normalized = reason.toLowerCase();
    return !normalized.includes("no description") &&
        !normalized.includes("description only contains practical information");
}

function getStableVariant(event, count) {
    const seed = `${event.id || ""}${event.name || ""}${event.date || ""}${event.city || ""}`;
    let hash = 0;

    for (let i = 0; i < seed.length; i += 1) {
        hash = (hash + seed.charCodeAt(i) * (i + 1)) % 9973;
    }

    return hash % count;
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}
