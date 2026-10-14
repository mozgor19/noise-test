# legacy/ — artık kullanılmayan dosyalar

Bu klasördeki hiçbir dosya güncel hatta kullanılmaz ve **makalede kullanılmamalıdır**.
Geçmişi görmek / eski sonuçlarla karşılaştırmak için saklanmıştır. Buradaki betikler
taşındıkları için göreli yolları artık çalışmayabilir.

| Klasör | İçerik | Neden geçersiz |
|---|---|---|
| `root_scripts/` | `evaluate_all_metrics.ipynb`, `evaluate_classification.ipynb`, `recompute_f1.py`, `dncnn_sp_noisy.py` (+ `dncnn_sp_noise.pth`), `emprcolor.m`, `adaptive_deneme.m` | Değerlendirme artık `evaluate.py` ile; notebook F1 maskesi kanal bazlı gürültüyü kaçırıyordu. `dncnn_sp_noisy.py` mavi kanalı hiç bozmuyordu. İki `.m` dosyasını hiçbir betik çağırmıyordu. |
| `old_results/` | Eski `results_*.csv`, PDF/HTML raporlar, `psnr_ssim_comparison.*`, `results_after_step2/`, `results_v3/` | DL sonuçları hatalı v2 modellerinden (≈ gürültülü girdi düzeyinde); SMF/MDBUTMF kenar hataları düzeltilmeden önce. |
| `test/`, `diagnostics/` | `full_eval_v2.py`, `run_tests.py`, teşhis betikleri ve çıktıları | v2 modelleri için yazıldı; SeConvUNet'e tamamı-1 maske veriyordu (sabit renk çıktısı). |
| `docs_examples_v1/` | README'nin ilk sürüm örnek şeritleri | README yalnız `docs/examples/v2/` kullanır. |
| `ai_based_models/configs/` | v1/v2 config'leri, harici model config'leri, v3 (sabit LR, 30 ep) ve v4 (kosinüs, 30 ep) config'leri | v1/v2: piksel bazlı gürültüyle eğitim. v3/v4: geçerli ama yakınsamamış ara sürümler; v4 makalede yalnızca yakınsama kontrolü. Güncel: `configs/*_v5.yaml`. |
| `ai_based_models/weights/` | v1/v2, v3, v4 ağırlıkları | v1/v2: test gürültüsüyle uyumsuz (`infer.py` reddeder). v3/v4: ara sürümler. |
| `ai_based_models/noisy_copies/` | `Tiny_Noisy_*` kopyaları | `Tiny-Imagenet/Tiny_Noisy_*` ile birebir aynı, kullanılmıyordu. |

Eski DL çıktı klasörleri veri dizinlerinin içinde duruyor: `Tiny-Imagenet/_legacy_v2/`, `480p-Test/_legacy_v2/` (geçersiz v2)
ve `Tiny-Imagenet/_legacy_v3/`, `480p-Test/_legacy_v3/` (v3; hocaya ilk gönderilen paragraf bu sonuçlara dayanıyordu).
