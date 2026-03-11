import tensorboard
from tensorboard import program
import os
from pathlib import Path

LOGDIR = r"C:\Users\Lukas\Desktop\ki_luca\3D-Verification\KI_Training_3_channels\trained_model\obj_verf_2cl_cnn_grid_big_new_thr_fold_0\logs"#os.path.join(os.getcwd(), "logs")

def launch_tensorboard(logdir=LOGDIR, port=6006):
    # optional: prüfen, ob der Ordner existiert
    assert Path(logdir).exists(), f"Logdir nicht gefunden: {logdir}"
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', str(port)])
    url = tb.launch()
    print(f"TensorBoard läuft unter: {url}")
    return url


if __name__ == "__main__":
    launch_tensorboard()