## Przed uruchomieniem
Do uruchomienia skryptów w projekcie należy zainstalować pakiety w pliku "requirements.txt" (albo "requirements_no_cuda.txt" w przypadku uruchamiania skryptu na komputerze bez GPU obsługującego technologię CUDA)

W celu uruchomienia środowiska gymnasium załączonego do projektu należy dodatkowo w folderze roms umieścić plik gry "Super JetPak DX" dostępnej do pobrania pod adresem https://asobitech.itch.io/super-jetpak-dx

## Opis najważniejszych parametrów modelu
model_path - nazwa folderu, w którym zostanie zapisany model
network - nazwa struktury sieci do użycia dla modelu
target_epoch - ilość epochów, na których będzie trenowany model
save_freq - co ile epochów model będzie zapisywany na dysku

## Trenowanie modelu
Do trenowania modelu służy skrypt training.py. Model zostanie wytrenowany zgodnie z parametramy zapisanymi w zmiennej "model_parameters"

## Uruchamianie wytrenowanego modelu
Do uruchomienia modelu służy skrypt play.py. Uruchomi on model znajdujący się w folderze o nazwie przekazanej do metody DQN.load()
