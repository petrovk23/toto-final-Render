files = ['Procfile','static/css/custom.css', 'static/js/main.js', 'templates/game_options.html', 'templates/combos.html', 'templates/index.html', 'templates/layout.html', 'templates/layout_no_sidebar.html', 'templates/results.html', 'analysis.py', 'app.py', 'database.py', 'init_database.py', 'config.py', 'requirements.txt', 'C_engine/src/analysis_engine.c', 'C_engine/src/analysis_engine.h', ]

# Open the file in write mode
with open('prompt_O1.txt', 'w') as output_file:

    output_file.write(f'<current_code>\n\n')

    # Loop through files and write their contents to prompt.txt
    for f in files:
        try:
            content = open(f).read().strip()
            output_file.write(f'{f}\n{content}\n\n')
        except FileNotFoundError:
            output_file.write(f'{f} not found\n\n')

    output_file.write(f'</current_code>\n\n')
    output_file.write(f'<goals>\n\n')
    output_file.write(f'</goals>\n\n')
    output_file.write(f'<constraints>\n\n')
    output_file.write(f'</constraints>\n\n')
    output_file.write(f'For any file you change, generate it end-to-end.\n\n')
