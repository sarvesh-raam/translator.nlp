document.addEventListener('DOMContentLoaded', () => {
    const sourceText = document.getElementById('source-text');
    const targetText = document.getElementById('target-text');
    const charCount = document.getElementById('curr-count');
    const loader = document.getElementById('loader');
    const heatmapContainer = document.getElementById('heatmap-container');

    let timeout = null;

    // --- Translation Logic ---
    sourceText.addEventListener('input', (e) => {
        const text = e.target.value;
        charCount.innerText = text.split(/\s+/).filter(x => x.length > 0).length;

        clearTimeout(timeout);
        timeout = setTimeout(() => {
            if (text.trim().length > 0) {
                translate(text);
            } else {
                targetText.innerHTML = '<p class="placeholder">Translation will appear here...</p>';
                heatmapContainer.innerHTML = '<div class="empty-viz">Translate something to see the attention weights</div>';
            }
        }, 800); // Debounce for 800ms
    });

    async function translate(text) {
        loader.classList.remove('hidden');
        try {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            const data = await response.json();
            
            if (data.translated) {
                targetText.innerHTML = `<p>${data.translated}</p>`;
                renderHeatmap(data.heatmap);
            }
        } catch (err) {
            console.error('Translation error:', err);
        } finally {
            loader.classList.add('hidden');
        }
    }

    // --- D3.js Heatmap Logic ---
    function renderHeatmap(data) {
        heatmapContainer.innerHTML = '';
        if (!data || data.length === 0) return;

        const margin = { top: 30, right: 20, bottom: 30, left: 60 },
              width = heatmapContainer.offsetWidth - margin.left - margin.right,
              height = 180 - margin.top - margin.bottom;

        const svg = d3.select("#heatmap-container")
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const sources = Array.from(new Set(data.map(d => d.source)));
        const targets = Array.from(new Set(data.map(d => d.target)));

        const x = d3.scaleBand().range([0, width]).domain(sources).padding(0.05);
        const y = d3.scaleBand().range([height, 0]).domain(targets).padding(0.05);

        const colorScale = d3.scaleSequential()
            .interpolator(d3.interpolatePurples)
            .domain([0, 1]);

        svg.selectAll()
            .data(data, d => d.source + ':' + d.target)
            .enter()
            .append("rect")
            .attr("x", d => x(d.source))
            .attr("y", d => y(d.target))
            .attr("width", x.bandwidth())
            .attr("height", y.bandwidth())
            .style("fill", d => colorScale(d.weight))
            .style("stroke-width", 4)
            .style("stroke", "none")
            .style("opacity", 0.8);

        // Add axes
        svg.append("g")
            .attr("transform", `translate(0, ${height})`)
            .call(d3.axisBottom(x).tickSize(0))
            .select(".domain").remove();

        svg.append("g")
            .call(d3.axisLeft(y).tickSize(0))
            .select(".domain").remove();
    }

    // --- Chart.js Metrics Logic ---
    async function loadMetrics() {
        try {
            const response = await fetch('/metrics');
            const data = await response.json();
            
            document.getElementById('final-bleu').innerText = data.bleu;
            document.getElementById('final-loss').innerText = data.training_history.loss.slice(-1)[0];

            const ctx = document.getElementById('metricsChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.training_history.epochs,
                    datasets: [{
                        label: 'Training Loss',
                        data: data.training_history.loss,
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { display: false },
                        y: { 
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { color: '#94a3b8' }
                        }
                    }
                }
            });
        } catch (err) {
            console.error('Metrics error:', err);
        }
    }

    // --- Initial Load ---
    loadMetrics();

    // --- Copy Logic ---
    function showTick(btn) {
        const originalText = btn.innerText;
        btn.innerText = '✅';
        btn.classList.add('success');
        setTimeout(() => {
            btn.innerText = originalText;
            btn.classList.remove('success');
        }, 2000);
    }

    document.getElementById('copy-source').onclick = (e) => {
        navigator.clipboard.writeText(sourceText.value);
        showTick(e.target);
    };

    document.getElementById('copy-target').onclick = (e) => {
        const text = targetText.innerText;
        if (text && !text.includes('Translation will appear here')) {
            navigator.clipboard.writeText(text);
            showTick(e.target);
        }
    };

    // --- Speaker Logic (TTS) ---
    const speakerBtn = document.getElementById('speaker-btn');
    speakerBtn.onclick = () => {
        const text = targetText.innerText;
        if (text && !text.includes('Translation will appear here')) {
            console.log("🔊 Speaking Hindi translation...");
            const utterance = new SpeechSynthesisUtterance(text);
            
            // Find a Hindi voice if available
            const voices = window.speechSynthesis.getVoices();
            const hindiVoice = voices.find(v => v.lang.includes('hi') || v.lang.includes('IN'));
            if (hindiVoice) utterance.voice = hindiVoice;
            
            utterance.lang = 'hi-IN';
            utterance.rate = 0.9; // Slightly slower for clarity
            window.speechSynthesis.speak(utterance);
            
            // Animation feedback
            speakerBtn.style.background = 'rgba(99, 102, 241, 0.4)';
            setTimeout(() => speakerBtn.style.background = '', 500);
        } else {
            console.warn("⚠️ Nothing to speak. Wait for translation...");
        }
    };
    
    // Pre-load voices (Chrome fix)
    window.speechSynthesis.getVoices();
});
