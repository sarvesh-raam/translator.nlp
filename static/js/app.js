document.addEventListener('DOMContentLoaded', () => {
    const sourceText = document.getElementById('source-text');
    const targetText = document.getElementById('target-text');
    const wordCount = document.getElementById('curr-count');
    const loader = document.getElementById('loader');
    const heatmapContainer = document.getElementById('heatmap-container');

    const PLACEHOLDER = 'अनुवाद यहाँ दिखेगा';
    const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

    let timeout = null;
    let lastHeatmap = null;
    let request = 0;

    // --- Translation ---
    function showTarget(text, isPlaceholder) {
        const p = document.createElement('p');
        // As text, not HTML: the model's "<unk>" token would otherwise be parsed as a tag and vanish
        p.textContent = text;
        if (isPlaceholder) p.className = 'placeholder';
        targetText.replaceChildren(p);
    }

    function showEmptyHeatmap() {
        lastHeatmap = null;
        heatmapContainer.innerHTML = '<div class="empty-viz">Translate a sentence to see its attention map.</div>';
    }

    // Grow the text box with its content instead of hiding the overflow
    function fitSource() {
        sourceText.style.height = 'auto';
        sourceText.style.height = sourceText.scrollHeight + 'px';
    }

    function onInput() {
        fitSource();
        const text = sourceText.value;
        wordCount.innerText = text.split(/\s+/).filter(x => x.length > 0).length;

        clearTimeout(timeout);
        timeout = setTimeout(() => {
            if (text.trim().length > 0) {
                translate(text);
            } else {
                showTarget(PLACEHOLDER, true);
                showEmptyHeatmap();
            }
        }, 600);
    }

    sourceText.addEventListener('input', onInput);

    document.querySelectorAll('.example').forEach(btn => {
        btn.addEventListener('click', () => {
            sourceText.value = btn.innerText;
            onInput();
            sourceText.focus();
        });
    });

    async function translate(text) {
        const id = ++request;
        loader.classList.remove('hidden');
        try {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            const data = await response.json();
            // Ignore answers to text that has since been edited
            if (id !== request) return;

            if (data.translated) {
                showTarget(data.translated, false);
                lastHeatmap = data.heatmap;
                renderHeatmap(data.heatmap);
            } else {
                showTarget('No translation: none of these words are in the model’s vocabulary.', true);
                showEmptyHeatmap();
            }
        } catch (err) {
            if (id === request) showTarget('Could not reach the server. Try again in a moment.', true);
            console.error('Translation error:', err);
        } finally {
            if (id === request) loader.classList.add('hidden');
        }
    }

    // --- Attention heatmap (D3) ---
    function renderHeatmap(data) {
        heatmapContainer.innerHTML = '';
        if (!data || data.length === 0) return;

        // Bands are word positions, not words, so a repeated word gets its own row/column
        const sourceWords = [], targetWords = [];
        data.forEach(d => {
            sourceWords[d.source_index] = d.source;
            targetWords[d.target_index] = d.target;
        });

        const cell = 34;
        const margin = { top: 8, right: 8, bottom: 70, left: 100 };
        const minWidth = sourceWords.length * cell;
        const width = Math.max(minWidth, Math.min(heatmapContainer.clientWidth - margin.left - margin.right,
                                                  sourceWords.length * 64));
        const height = targetWords.length * cell;

        const svg = d3.select('#heatmap-container')
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const x = d3.scaleBand().range([0, width]).domain(d3.range(sourceWords.length)).padding(0.06);
        // First output word on top, so rows read in translation order
        const y = d3.scaleBand().range([0, height]).domain(d3.range(targetWords.length)).padding(0.06);
        const color = d3.scaleLinear().domain([0, 1]).range([css('--surface'), css('--accent')]).clamp(true);

        svg.selectAll('rect')
            .data(data, d => d.source_index + ':' + d.target_index)
            .enter()
            .append('rect')
            .attr('x', d => x(d.source_index))
            .attr('y', d => y(d.target_index))
            .attr('width', x.bandwidth())
            .attr('height', y.bandwidth())
            .style('fill', d => color(d.weight))
            .style('stroke', css('--rule'))
            .style('stroke-width', 0.5)
            .append('title')
            .text(d => `${d.target} ← ${d.source}: ${(d.weight * 100).toFixed(0)}%`);

        svg.append('g')
            .attr('transform', `translate(0, ${height})`)
            .call(d3.axisBottom(x).tickSize(0).tickPadding(8).tickFormat(i => sourceWords[i]))
            .call(g => g.select('.domain').remove())
            .selectAll('text')
            .attr('text-anchor', 'end')
            .attr('transform', 'rotate(-40)')
            .attr('dx', '-0.3em')
            .attr('dy', '0.4em');

        svg.append('g')
            .attr('class', 'target-axis')
            .call(d3.axisLeft(y).tickSize(0).tickPadding(10).tickFormat(i => targetWords[i]))
            .call(g => g.select('.domain').remove());
    }

    let resizeTimer = null;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            fitSource();
            if (lastHeatmap) renderHeatmap(lastHeatmap);
        }, 150);
    });

    // --- Training metrics (Chart.js) ---
    async function loadMetrics() {
        try {
            const response = await fetch('/metrics');
            const data = await response.json();
            const history = data.training_history;

            document.getElementById('final-bleu').innerText = Number(data.bleu).toFixed(2);
            document.getElementById('epoch-count').innerText = history.epochs.length;
            document.getElementById('final-loss').innerText = history.loss.slice(-1)[0].toFixed(2);
            if (history.val_bleu) {
                const best = Math.max(...history.val_bleu);
                const bestEpoch = history.val_bleu.indexOf(best) + 1;
                document.getElementById('best-val-bleu').innerText = best.toFixed(2);
                document.getElementById('metrics-desc').innerText =
                    `Training loss and validation BLEU over ${history.epochs.length} epochs. ` +
                    `Training stopped after three epochs without a better validation score; ` +
                    `the weights from epoch ${bestEpoch} are the ones serving this page.`;
            }

            const ink = css('--ink-soft');
            const faint = css('--ink-faint');
            const rule = css('--rule');
            const accent = css('--accent');
            Chart.defaults.font.family = css('--mono');
            Chart.defaults.font.size = 11;
            Chart.defaults.color = faint;

            const datasets = [{
                label: 'Training loss',
                data: history.loss,
                borderColor: ink,
                borderWidth: 1.5,
                pointRadius: 2,
                pointBackgroundColor: ink,
                tension: 0.25,
                yAxisID: 'y'
            }];
            if (history.val_bleu) {
                datasets.push({
                    label: 'Validation BLEU',
                    data: history.val_bleu,
                    borderColor: accent,
                    borderWidth: 1.5,
                    pointRadius: 2,
                    pointBackgroundColor: accent,
                    tension: 0.25,
                    yAxisID: 'y1'
                });
            }

            new Chart(document.getElementById('metricsChart'), {
                type: 'line',
                data: { labels: history.epochs, datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: {
                            align: 'start',
                            labels: { color: ink, boxWidth: 14, boxHeight: 1, padding: 16 }
                        },
                        tooltip: {
                            callbacks: {
                                title: items => `Epoch ${items[0].label}`,
                                label: item => `${item.dataset.label}: ${item.parsed.y.toFixed(2)}`
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'epoch', color: faint },
                            grid: { display: false },
                            border: { color: rule }
                        },
                        y: {
                            title: { display: true, text: 'loss', color: ink },
                            grid: { color: rule },
                            border: { display: false }
                        },
                        y1: {
                            display: datasets.length > 1,
                            position: 'right',
                            title: { display: true, text: 'BLEU', color: accent },
                            ticks: { color: accent },
                            grid: { drawOnChartArea: false },
                            border: { display: false }
                        }
                    }
                }
            });
        } catch (err) {
            console.error('Metrics error:', err);
        }
    }

    loadMetrics();

    // --- Copy and listen ---
    function flash(btn, text) {
        const original = btn.dataset.label || btn.innerText;
        btn.dataset.label = original;
        btn.innerText = text;
        btn.classList.add('done');
        setTimeout(() => {
            btn.innerText = original;
            btn.classList.remove('done');
        }, 1500);
    }

    function translation() {
        const p = targetText.querySelector('p');
        return p && !p.classList.contains('placeholder') ? p.textContent : '';
    }

    document.getElementById('copy-source').onclick = (e) => {
        if (!sourceText.value) return;
        navigator.clipboard.writeText(sourceText.value);
        flash(e.target, 'Copied');
    };

    document.getElementById('copy-target').onclick = (e) => {
        const text = translation();
        if (!text) return;
        navigator.clipboard.writeText(text);
        flash(e.target, 'Copied');
    };

    const speakerBtn = document.getElementById('speaker-btn');
    speakerBtn.onclick = () => {
        const text = translation();
        if (!text || !window.speechSynthesis) return;

        const utterance = new SpeechSynthesisUtterance(text.replace(/<unk>/g, ''));
        const hindiVoice = window.speechSynthesis.getVoices().find(v => v.lang.startsWith('hi'));
        if (hindiVoice) utterance.voice = hindiVoice;
        utterance.lang = 'hi-IN';
        utterance.rate = 0.9;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
        flash(speakerBtn, 'Playing');
    };

    // Chrome loads voices asynchronously
    if (window.speechSynthesis) window.speechSynthesis.getVoices();
});
