using System.Diagnostics;
using System.Globalization;
using System.Text.Json;

// Monitor do treino GRPO/SFT: visao LIVE + HISTORICO (H alterna, Q sai).
// Uso: monitor.exe [logJson] [pollSeg]
var C = CultureInfo.InvariantCulture;
var exeDir = AppContext.BaseDirectory;
var logPath = args.Length > 0 ? args[0]
    : Path.GetFullPath(Path.Combine(exeDir, "..", "data", "tool_grpo_v3_training_log.json"));
var poll = args.Length > 1 && int.TryParse(args[1], out var p) ? Math.Max(1, p) : 2;
var mode = 0; // 0 live, 1 historico
Console.WriteLine($"monitor: {logPath} a cada {poll}s (H historico | Q sai)");

bool QuitPressed(out bool hist)
{
    hist = false;
    try
    {
        if (!Console.KeyAvailable) return false;
        var k = Console.ReadKey(true).Key;
        if (k == ConsoleKey.Q) return true;
        if (k == ConsoleKey.H) hist = true;
        return false;
    }
    catch { return false; }
}

while (true)
{
    if (QuitPressed(out var h1)) break;
    if (h1) mode = 1 - mode;
    try
    {
        if (mode == 0) RenderLive(logPath);
        else RenderHist(logPath);
    }
    catch (Exception e) { Console.WriteLine("erro: " + e.Message); }
    for (int i = 0; i < poll * 2; i++)
    {
        if (QuitPressed(out var h2)) goto fim;
        if (h2) { mode = 1 - mode; break; }
        Thread.Sleep(500);
    }
}
fim:;

List<JsonElement> Load(string path)
{
    using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
    return JsonSerializer.Deserialize<List<JsonElement>>(fs) ?? new();
}

double G(JsonElement e, string k) =>
    e.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : double.NaN;
int GI(JsonElement e, string k) =>
    e.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : 0;

string Bars(List<double> vals, int w)
{
    const string scale = " .:-=+*#@";
    if (vals.Count == 0) return "";
    var mn = vals.Min(); var mx = vals.Max(); var rg = Math.Max(1e-9, mx - mn);
    var sampled = vals.Count <= w ? vals
        : vals.Where((_, i) => i % ((vals.Count + w - 1) / w) == 0).ToList();
    return new string(sampled.Select(v => scale[(int)((v - mn) / rg * (scale.Length - 1))]).ToArray())
           + $"  min {mn.ToString("F2", C)} max {mx.ToString("F2", C)}";
}

void RenderLive(string path)
{
    Console.Clear();
    Console.WriteLine("=== QWEN3LOOP TRAIN MONITOR [LIVE - H historico] ===");
    Console.WriteLine(DateTime.Now.ToString("HH:mm:ss", C) + "  " + path);
    var alerts = new List<string>();
    if (!File.Exists(path)) { Console.WriteLine("log ainda nao existe."); return; }
    var stale = DateTime.Now - File.GetLastWriteTime(path);
    if (stale.TotalSeconds > 180) alerts.Add($"LOG STALE ha {stale.TotalMinutes:F0}min (travou/caiu?)");

    List<JsonElement> hist;
    try { hist = Load(path); }
    catch { Console.WriteLine("lendo log (parcial)..."); return; }
    if (hist.Count == 0) { Console.WriteLine("log vazio."); return; }

    var last = hist[^1];
    var step = GI(last, "step");
    Console.WriteLine($"step {step}  fase {GI(last, "phase")}  reward {G(last, "reward").ToString("F3", C)}  hits {GI(last, "hits")}/4");
    var nc = G(last, "nocall_frac");
    Console.WriteLine($"PPL-L2 {G(last, "ppl_loop2").ToString("F2", C)}  PPL-L1 {G(last, "ppl_loop1").ToString("F2", C)}  NoCall {(double.IsNaN(nc) ? "?" : nc.ToString("F2", C))}");
    Console.WriteLine($"VRAM pico(log) {G(last, "peak_vram_gb").ToString("F2", C)} GB  t_step {G(last, "step_time_sec").ToString("F1", C)}s  ETA {G(last, "eta_min").ToString("F1", C)}min");

    var tail = hist.Skip(Math.Max(0, hist.Count - 20)).Select(e => G(e, "reward")).Where(double.IsFinite).ToList();
    if (tail.Count > 1)
    {
        Console.WriteLine("reward[20]: " + Bars(tail, 40));
        if (tail.Count >= 5 && tail.Skip(tail.Count - 5).All(v => Math.Abs(v - tail[^1]) < 1e-9))
            alerts.Add("reward congelado nos ultimos 5 steps");
    }
    var ppl = G(last, "ppl_loop2");
    if (!double.IsNaN(ppl) && ppl > 50) alerts.Add($"PPL-L2 ALTA: {ppl.ToString("F1", C)} (risco verborragia/divergencia)");
    if (!double.IsNaN(nc) && nc > 0.6) alerts.Add($"NoCall alto: {nc.ToString("F2", C)} (risco over-abstencao)");

    try
    {
        var psi = new ProcessStartInfo("nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits")
        { RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
        using var pr = Process.Start(psi);
        var line = (pr?.StandardOutput.ReadToEnd() ?? "").Split('\n', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault() ?? "";
        pr?.WaitForExit(5000);
        var f = line.Split(',').Select(s => s.Trim()).ToArray();
        if (f.Length >= 5)
        {
            Console.WriteLine($"GPU: util {f[0]}%  VRAM {f[1]}/{f[2]} MB  temp {f[3]}C  power {f[4]}W");
            if (int.TryParse(f[3], out var t) && t > 83) alerts.Add($"GPU quente: {t}C");
            if (double.TryParse(f[1], out var mu) && double.TryParse(f[2], out var mt) && mt > 0 && mu / mt > 0.92)
                alerts.Add("VRAM > 92% (risco OOM)");
        }
    }
    catch { Console.WriteLine("nvidia-smi indisponivel."); }

    Console.WriteLine(alerts.Count == 0 ? "[OK] sem alertas" : "[ALERTAS]");
    foreach (var a in alerts) Console.WriteLine("  ! " + a);
}

void RenderHist(string path)
{
    Console.Clear();
    Console.WriteLine("=== QWEN3LOOP TRAIN MONITOR [HISTORICO - H live] ===");
    List<JsonElement> hist;
    try { hist = Load(path); }
    catch { Console.WriteLine("lendo log (parcial)..."); return; }
    if (hist.Count == 0) { Console.WriteLine("log vazio."); return; }
    Console.WriteLine($"steps: {hist.Count}  fases: {string.Join(",", hist.Select(e => GI(e, "phase")).Distinct().OrderBy(x => x))}");

    var rw = hist.Select(e => G(e, "reward")).Where(double.IsFinite).ToList();
    var p2 = hist.Select(e => G(e, "ppl_loop2")).Where(double.IsFinite).ToList();
    var ht = hist.Select(e => (double)GI(e, "hits")).ToList();
    var ncl = hist.Select(e => G(e, "nocall_frac")).Where(double.IsFinite).ToList();
    Console.WriteLine("reward : " + Bars(rw, 60));
    Console.WriteLine("PPL-L2 : " + Bars(p2, 60));
    Console.WriteLine("hits/4 : " + Bars(ht, 60));
    if (ncl.Count > 0) Console.WriteLine("NoCall : " + Bars(ncl, 60));
    if (rw.Count > 0)
    {
        var t = Math.Max(1, rw.Count / 3);
        var m1 = rw.Take(t).Average(); var m3 = rw.Skip(rw.Count - t).Average();
        Console.WriteLine($"media reward: total {rw.Average().ToString("F2", C)} | 1o terco {m1.ToString("F2", C)} | ult terco {m3.ToString("F2", C)} | tendencia {(m3 >= m1 ? "ALTA" : "QUEDA")}");
    }
    var tot = hist.Sum(e => G(e, "step_time_sec") is double s && !double.IsNaN(s) ? s : 0);
    Console.WriteLine($"tempo decorrido: {(tot / 60).ToString("F1", C)}min  ultimo ETA: {G(hist[^1], "eta_min").ToString("F1", C)}min");
}
