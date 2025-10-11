import os
import yaml
def create_new_config():
    def rename_recursive(d,inname):
        if isinstance(d,dict):
            for key,val in d.items():
                if isinstance(d[key], str):
                    d[key]=d[key].replace("<NAME>", inname)
                else:
                    rename_recursive(d[key],inname)
        return d
    base_conf_path = "configs/train"    
    print(f"Пропиши имя файла конфига в папке {base_conf_path}, например: 'divertito.yaml'")
    main_conf_name = str(input()).strip()
    if not main_conf_name.endswith(".yaml"):
        main_conf_name +=".yaml"
    main_conf_path = os.path.join(base_conf_path,main_conf_name)
    while not os.path.exists(main_conf_path):
        print(f"Конфиг {main_conf_path} не существует! Создать новый? (да/нет/выйти)")
        inp2 = str(input()).strip().lower()
        if inp2=="да":
            print("Создаю новый конфиг из дефолтного...")
            default_conf = yaml.safe_load(open(f"{base_conf_path}/default.yaml"))
            """1"""
            print("(Шаг 1/): Введите имя спикера (Например, divertito):")
            inp3= str(input()).strip().lower()
            default_conf=rename_recursive(default_conf, inp3)
            """2"""
            main_dir_path=default_conf["data"]["main_dir"]
            os.makedirs(main_dir_path, exist_ok=True)
            print(f"Создаю путь до главной папки с файлами: {main_dir_path}")
            """3"""
            weights_folder=default_conf["data"]["weights_folder"]
            os.makedirs(weights_folder, exist_ok=True)
            print(f"Создаю путь до папки сохранения обученных весов {weights_folder}")
            """4"""
            print("(Шаг 2/): Введите путь до .wav файла для обучения:")
            inp4= str(input()).strip()
            while not os.path.exists(inp4):
                print(f"(Шаг 2/): Путь {inp4} не существует! Введите корретный путь до .wav файла:")
                inp4= str(input()).strip()
            default_conf["data"]["base_wav_path"] = inp4
            os.makedirs(os.path.dirname(inp4), exist_ok=True)
            os.makedirs(os.path.join(os.path.dirname(inp4),"wavs/"), exist_ok=True)
            main_conf_path = os.path.join(base_conf_path, f"{inp3}.yaml")
            print(f"Конфиг '{inp3}.yaml' сохранен в {base_conf_path}!")
    
            yaml.safe_dump(default_conf, open(main_conf_path, "w",  encoding='utf-8'),
                            default_flow_style=False,  # Let our custom representer decide
                            sort_keys=False,
                            width=2000,
                            allow_unicode=True
                        )  # Prevent line wrapping)
            break

        elif inp2=="нет":
            print("Пропиши ссылку до конфига, например: 'configs/main/divertito.yaml' .")
        elif inp2=="выйти":
            print("Выхожу из создания конфига...")
            break
        else:
            print(f"Неизвестный ответ {inp2}!")
            continue

create_new_config()