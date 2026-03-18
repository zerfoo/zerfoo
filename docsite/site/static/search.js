(function() {
    "use strict";

    var searchInput = document.getElementById("search-input");
    var searchResults = document.getElementById("search-results");
    var searchIndex = null;

    function loadIndex() {
        if (searchIndex !== null) return;
        var xhr = new XMLHttpRequest();
        xhr.open("GET", "/search-index.json", true);
        xhr.onload = function() {
            if (xhr.status === 200) {
                searchIndex = JSON.parse(xhr.responseText);
            }
        };
        xhr.send();
    }

    function search(query) {
        if (!searchIndex || query.length < 2) {
            searchResults.classList.remove("visible");
            return;
        }

        var terms = query.toLowerCase().split(/\s+/);
        var results = [];

        for (var i = 0; i < searchIndex.length; i++) {
            var entry = searchIndex[i];
            var titleLower = entry.title.toLowerCase();
            var bodyLower = entry.body.toLowerCase();
            var score = 0;

            for (var t = 0; t < terms.length; t++) {
                var term = terms[t];
                if (titleLower.indexOf(term) !== -1) {
                    score += 10;
                }
                if (bodyLower.indexOf(term) !== -1) {
                    score += 1;
                }
            }

            if (score > 0) {
                results.push({ entry: entry, score: score });
            }
        }

        results.sort(function(a, b) { return b.score - a.score; });
        results = results.slice(0, 10);

        if (results.length === 0) {
            searchResults.classList.remove("visible");
            return;
        }

        var html = "";
        for (var r = 0; r < results.length; r++) {
            var res = results[r].entry;
            var snippet = getSnippet(res.body, terms[0]);
            html += '<a href="' + res.path + '">' +
                '<div>' + escapeHTML(res.title) + '</div>' +
                '<div class="search-snippet">' + escapeHTML(snippet) + '</div>' +
                '</a>';
        }

        searchResults.innerHTML = html;
        searchResults.classList.add("visible");
    }

    function getSnippet(body, term) {
        var idx = body.toLowerCase().indexOf(term);
        if (idx === -1) idx = 0;
        var start = Math.max(0, idx - 40);
        var end = Math.min(body.length, idx + 80);
        var snippet = body.substring(start, end).trim();
        if (start > 0) snippet = "..." + snippet;
        if (end < body.length) snippet = snippet + "...";
        return snippet;
    }

    function escapeHTML(s) {
        var div = document.createElement("div");
        div.textContent = s;
        return div.innerHTML;
    }

    searchInput.addEventListener("focus", loadIndex);
    searchInput.addEventListener("input", function() {
        search(this.value);
    });

    document.addEventListener("click", function(e) {
        if (!searchResults.contains(e.target) && e.target !== searchInput) {
            searchResults.classList.remove("visible");
        }
    });
})();
