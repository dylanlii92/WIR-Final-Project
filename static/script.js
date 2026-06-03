const searchForm = document.getElementById("search-form");
const searchInput = document.getElementById("search-input");
const resultsContainer = document.getElementById("results");
const statusContainer = document.getElementById("status");
const showMoreButton = document.getElementById("show-more");

const RESULTS_PER_PAGE = 6;
const INITIAL_RESULTS_LIMIT = 12;
let currentQuery = "";
let nextOffset = 0;
let visibleEventCount = 0;
let isLoadingMore = false;
let bestGrid = null;
let alsoLikeGrid = null;

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
    bestGrid = null;
    alsoLikeGrid = null;

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
    const limit = append ? RESULTS_PER_PAGE : INITIAL_RESULTS_LIMIT;
    const params = new URLSearchParams({
        q: currentQuery,
        limit,
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

    let renderedCount = events.length;

    if (append) {
        appendEventsToGrid(events, getOrCreateAlsoLikeSection());
    } else {
        renderedCount = renderInitialSections(events);
    }

    visibleEventCount += renderedCount;
    nextOffset = visibleEventCount;
    statusContainer.textContent = `Showing ${visibleEventCount} events`;
    showMoreButton.hidden = !data.has_more && events.length <= renderedCount;
}

function renderInitialSections(events) {
    const bestCount = getBestRecommendationCount(events);
    const bestEvents = events.slice(0, bestCount);
    const alsoLikeEvents = events.slice(bestCount, bestCount + RESULTS_PER_PAGE);

    if (bestEvents.length) {
        bestGrid = createResultsSection("Our best recommendations");
        appendEventsToGrid(bestEvents, bestGrid);
    }

    if (alsoLikeEvents.length) {
        alsoLikeGrid = createResultsSection("You may also like");
        appendEventsToGrid(alsoLikeEvents, alsoLikeGrid);
    }

    return bestEvents.length + alsoLikeEvents.length;
}

function createResultsSection(title) {
    const section = document.createElement("section");
    section.className = "results-group";

    section.innerHTML = `
        <h2 class="results-heading">${escapeHtml(title)}</h2>
        <div class="results-grid"></div>
    `;

    resultsContainer.appendChild(section);

    return section.querySelector(".results-grid");
}

function getOrCreateAlsoLikeSection() {
    if (!alsoLikeGrid) {
        alsoLikeGrid = createResultsSection("You may also like");
    }

    return alsoLikeGrid;
}

function appendEventsToGrid(events, grid) {
    events.forEach(function (event) {
        grid.appendChild(createEventCard(event));
    });
}

function getBestRecommendationCount(events) {
    const strongMatches = events.filter(function (event) {
        return event.artist_query_match && event.city_query_match;
    }).length;

    if (strongMatches >= 6) {
        return 6;
    }

    return Math.min(3, events.length);
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

            <p><strong>Artist:</strong> ${escapeHtml(event.artist || "Unknown")}</p>

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
    const bonuses = event.score_breakdown && event.score_breakdown.bonuses
        ? event.score_breakdown.bonuses
        : {};

    if (event.artist_query_match || bonuses.artist_match) {
        labels.push("Matched artist");
    }

    if (event.city_query_match || bonuses.city_match) {
        labels.push("Matched city");
    }

    if (bonuses.state_match) {
        labels.push("Matched state");
    }

    if (bonuses.title_match) {
        labels.push("Matched event title");
    }

    if (bonuses.genre_match) {
        labels.push("Matched genre");
    }

    if (bonuses.subgenre_match) {
        labels.push("Matched subgenre");
    }

    if (bonuses.date_window_match) {
        labels.push("Matched date range");
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