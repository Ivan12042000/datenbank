import tkinter as tk
from tkinter import ttk
import pandas as pd
import os
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global placeholder for Eingabefeldern
eingabefelder = {}

# Funktion: Lade letzte 5 Einträge aller Tabellen für Tabellenansicht
def lade_tabellen_automatisch():
    skriptpfad = os.path.dirname(os.path.abspath(__file__))
    laden = {}
    for name in ['Wochentage', 'Veraenderung_durchschnitt', 'Veraenderung_unterschied', 'prozent_durchschnitt', 'prozent_unterschied']:
        pfad = os.path.join(skriptpfad, f"{name}.csv")
        if os.path.exists(pfad):
            df = pd.read_csv(pfad, engine='python').dropna(how='all')
            if 'Woche' in df.columns:
                try:
                    df['Woche_num'] = pd.to_numeric(df['Woche'], errors='coerce')
                    df.sort_values(by='Woche_num', inplace=True)
                except:
                    df.sort_values(by='Woche', inplace=True)
                df = df.tail(5).copy()
                df.drop(columns=['Woche_num'], inplace=True, errors='ignore')
            laden[name] = df
        else:
            laden[name] = pd.DataFrame()
    return laden

# Funktion: Berechne neue Wochentage-Zeile
def berechne_naechste_woche():
    skriptpfad = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(skriptpfad, 'Wochentage.csv')
    df = pd.read_csv(filepath, engine='python').dropna(how='all')
    df['Enddatum'] = pd.to_datetime(df['Enddatum'], format='%Y-%m-%d', errors='coerce')
    letzte_woche = df['Woche'].astype(int).max()
    letzte_subs = df.loc[df['Woche'] == letzte_woche, 'Subs'].iloc[0]
    letzte_end = df.loc[df['Woche'] == letzte_woche, 'Enddatum'].iloc[0]
    naechste_woche = letzte_woche + 1
    anfangsdatum = letzte_end + pd.Timedelta(days=1)
    enddatum = anfangsdatum + pd.Timedelta(days=6)
    # Werte aus Eingabe
    rohdaten = {}
    for tag in ['Montag','Dienstag','Mittwoch','Donnerstag','Freitag','Samstag','Sonntag']:
        val = eingabefelder.get(tag).get() if tag in eingabefelder else '0'
        try: wert = int(val)
        except: wert = 0
        rohdaten[tag.lower()] = wert
    gesamt = sum(rohdaten.values())
    proz = {f'proz_{tag.lower()}': round((wert/gesamt*100) if gesamt>0 else 0, 3)
            for tag, wert in rohdaten.items()}
    neue_subs = letzte_subs + gesamt
    return pd.DataFrame([{  
        'Woche': naechste_woche,
        'Anfangsdatum': anfangsdatum.strftime('%Y-%m-%d'),
        'Enddatum': enddatum.strftime('%Y-%m-%d'),
        **rohdaten, **proz,
        'Gesamt': gesamt,
        'Subs': neue_subs
    }])

# Funktion: Anzeige DataFrame in Treeview
def zeige_tabelle(frame, df):
    for w in frame.winfo_children(): w.destroy()
    if df is None or df.empty:
        ttk.Label(frame, text='Keine Daten vorhanden.').pack(pady=20)
        return
    cont = ttk.Frame(frame); cont.pack(expand=True, fill='both')
    cols = list(df.columns)
    tree = ttk.Treeview(cont, columns=cols, show='headings', height=8)
    tree.pack(side='top', expand=True, fill='both')
    sb = ttk.Scrollbar(cont, orient='horizontal', command=tree.xview)
    sb.pack(side='bottom', fill='x'); tree.configure(xscrollcommand=sb.set)
    for c in cols:
        tree.heading(c, text=c); tree.column(c, anchor='center')
    for _, row in df.iterrows():
        tree.insert('', 'end', values=list(row))

# Funktion: Analyse-Tab mit Regression und Prognosetabelle
def zeige_analyse(frame):
    for w in frame.winfo_children():
        w.destroy()
    skriptpfad = os.path.dirname(os.path.abspath(__file__))
    pfad = os.path.join(skriptpfad, 'Wochentage.csv')
    if not os.path.exists(pfad):
        ttk.Label(frame, text='Wochentage.csv nicht gefunden.').pack(pady=20)
        return
    df_full = pd.read_csv(pfad, engine='python').dropna(how='all')
    if 'Woche' not in df_full.columns or 'Subs' not in df_full.columns:
        ttk.Label(frame, text='Spalten "Woche" oder "Subs" fehlen.').pack(pady=20)
        return
    # Regression über alle Daten
    x = pd.to_numeric(df_full['Woche'], errors='coerce')
    y = pd.to_numeric(df_full['Subs'], errors='coerce')
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    # Prognoseziele: 10 Schritte ab letzter Subs, in 10k Schritten
    last_subs = y.iloc[-1]
    start_val = np.ceil(last_subs / 10000) * 10000
    targets = start_val + np.arange(0, 10) * 10000
    # Prognosewochen
    w_pred = (targets - coeffs[1]) / coeffs[0]
    # Plotbereich bis letzte prognostizierte Woche
    max_week = w_pred.max()
    x_line = np.linspace(x.min(), max_week, 100)
    y_line = poly(x_line)
    # Erzeugen des Plots
    fig = Figure(figsize=(6,4), dpi=100)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, label='Datenpunkte')
    ax.plot(x_line, y_line, label=f'Regression: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}')
    ax.scatter(w_pred, targets, color='red', label='Prognosepunkte')
    ax.set_title('Regression Subs vs. Woche (alle Daten)')
    ax.set_xlabel('Woche')
    ax.set_ylabel('Subs')
    ax.legend()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(expand=True, fill='both')
    # Prognosetabelle
    table_frame = ttk.Frame(frame)
    table_frame.pack(pady=10)
    cols = ['Erreichte Subs', 'Woche']
    tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=len(targets))
    tree.pack(side='left', expand=True, fill='both')
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, anchor='center')
    for val, wk in zip(targets, w_pred):
        tree.insert('', 'end', values=[int(val), f"{wk:.0f}"])
    vsb = ttk.Scrollbar(table_frame, orient='vertical', command=tree.yview)
    vsb.pack(side='right', fill='y')
    tree.configure(yscrollcommand=vsb.set)

# Hauptprogramm
if __name__ == '__main__':
    other_tables = lade_tabellen_automatisch()
    root = tk.Tk(); root.title('Meine Auswertungs-GUI'); root.geometry('1100x850')
    main_nb = ttk.Notebook(root); main_nb.pack(fill='both', expand=True)
    # 1) Tabellenansicht
    tab_frame = ttk.Frame(main_nb); main_nb.add(tab_frame, text='Tabellenansicht')
    sub_nb = ttk.Notebook(tab_frame); sub_nb.pack(fill='both', expand=True)
    for name, df in other_tables.items():
        f = ttk.Frame(sub_nb); sub_nb.add(f, text=name); zeige_tabelle(f, df)
    # 2) Dateneingabe
    input_frame = ttk.Frame(main_nb); main_nb.add(input_frame, text='Dateneingabe')
    # Header berechnen und anzeigen
    df_h = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Wochentage.csv'), engine='python').dropna(how='all')
    df_h['Enddatum'] = pd.to_datetime(df_h['Enddatum'], format='%Y-%m-%d', errors='coerce')
    last_week = df_h['Woche'].astype(int).max()
    start_date = df_h.loc[df_h['Woche'] == last_week, 'Enddatum'].iloc[0] + pd.Timedelta(days=1)
    end_date = start_date + pd.Timedelta(days=6)
    header_label = tk.Label(
        input_frame,
        text=f"Eingabe für Woche {last_week+1} im Zeitraum von {start_date.strftime('%Y-%m-%d')} bis {end_date.strftime('%Y-%m-%d')}",
        font=('Helvetica', 14, 'bold'),
        pady=10
    )
    header_label.pack()
    # Eingabefelder für jeden Wochentag
    eingabe_frame = ttk.Frame(input_frame)
    eingabe_frame.pack(pady=20)
    for tag in ['Montag','Dienstag','Mittwoch','Donnerstag','Freitag','Samstag','Sonntag']:
        row = ttk.Frame(eingabe_frame)
        row.pack(fill='x', pady=5)
        ttk.Label(row, text=tag, width=15, anchor='w').pack(side='left')
        entry = ttk.Entry(row, width=50)
        entry.pack(side='left', expand=True, fill='x')
        eingabefelder[tag] = entry
    # Vorschau-Button
    def open_vorschau():
        win = tk.Toplevel(root); win.title('Vorschau')
        nb = ttk.Notebook(win); nb.pack(fill='both', expand=True)
        # Wochentage-Tab
        f1 = ttk.Frame(nb); nb.add(f1, text='Wochentage')
        df_orig = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Wochentage.csv'), engine='python').dropna(how='all')
        try: df_orig['Woche_num']=pd.to_numeric(df_orig['Woche'],errors='coerce'); df_orig.sort_values('Woche_num', inplace=True)
        except: df_orig.sort_values('Woche', inplace=True)
        df_last4 = df_orig.tail(4).copy(); df_new = berechne_naechste_woche()
        df_w_preview = pd.concat([df_last4, df_new], ignore_index=True)
        cols_w = ['Woche','Anfangsdatum','Enddatum'] + [c for pair in zip(
            ['montag','dienstag','mittwoch','donnerstag','freitag','samstag','sonntag'],
            [f'proz_{d}' for d in ['montag','dienstag','mittwoch','donnerstag','freitag','samstag','sonntag']]
        ) for c in pair] + ['Gesamt','Subs']
        df_w_preview = df_w_preview[cols_w]
        zeige_tabelle(f1, df_w_preview)
                        # Veränderung_durchschnitt-Tab
        f2 = ttk.Frame(nb); nb.add(f2, text='Veränderung_durchschnitt')
        df_avg = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Veraenderung_durchschnitt.csv'), engine='python').dropna(how='all')
        # Sortieren
        try:
            df_avg['Woche_num'] = pd.to_numeric(df_avg['Woche'], errors='coerce')
            df_avg.sort_values('Woche_num', inplace=True)
        except:
            df_avg.sort_values('Woche', inplace=True)
        df_last4_avg = df_avg.tail(4).copy()
        # Entferne interne Spalte 'Woche_num' für saubere Vorschau
        if 'Woche_num' in df_last4_avg.columns:
            df_last4_avg.drop(columns=['Woche_num'], inplace=True)
        # Berechne neue Durchschnittswerte aus Wochentage.csv + Eingaben
        df_w = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Wochentage.csv'), engine='python').dropna(how='all')
        neue_werte = {}
        for tag in ['Montag','Dienstag','Mittwoch','Donnerstag','Freitag','Samstag','Sonntag']:
            hist_vals = pd.to_numeric(df_w[tag.lower()], errors='coerce').fillna(0).tolist()
            try:
                hist_vals.append(int(eingabefelder[tag].get()))
            except:
                hist_vals.append(0)
            neue_werte[tag.lower()] = round(sum(hist_vals) / len(hist_vals), 3)
        gesamt_neu = round(sum(neue_werte.values()), 3)
        last_week_num = df_last4_avg['Woche'].astype(int).max()
        # Mappe neue_werte in passende Spaltennamen (Großschreibung)
        new_row = {'Woche': last_week_num + 1}
        for key, val in neue_werte.items():
            col_name = key.capitalize()  # montag -> Montag
            new_row[col_name] = val
        new_row['Gesamt'] = gesamt_neu
                        # Erstelle kombiniertes DataFrame mit letzter 4 Zeilen und neuer Zeile
        df_avg_new = pd.DataFrame([new_row], columns=df_last4_avg.columns)
        df_avg_preview = pd.concat([df_last4_avg, df_avg_new], ignore_index=True)
        zeige_tabelle(f2, df_avg_preview)

                        # Veränderung_unterschied-Tab: letzte 4 Einträge + neue Diff-Zeile
        f3 = ttk.Frame(nb); nb.add(f3, text='Veränderung_unterschied')
        # Lade tatsächliche Veränderung_unterschied.csv
        df_diff = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Veraenderung_unterschied.csv'), engine='python').dropna(how='all')
        # Sortieren
        try:
            df_diff['Woche_num'] = pd.to_numeric(df_diff['Woche'], errors='coerce')
            df_diff.sort_values('Woche_num', inplace=True)
        except:
            df_diff.sort_values('Woche', inplace=True)
        df_last4_diff = df_diff.tail(4).copy()
        if 'Woche_num' in df_last4_diff.columns:
            df_last4_diff.drop(columns=['Woche_num'], inplace=True)
        # Berechne neue Unterschiede: Differenz zur letzten Durchschnittsberechnung
        diff_row = {'Woche': new_row['Woche']}
        prev = df_last4_avg.iloc[-1]
        for tag in ['Montag','Dienstag','Mittwoch','Donnerstag','Freitag','Samstag','Sonntag']:
            val_new = new_row[tag]
            val_prev = prev[tag]
            diff_row[tag] = round(val_new - val_prev, 3)
        diff_row['Gesamt'] = round(sum(diff_row[tag] for tag in ['Montag','Dienstag','Mittwoch','Donnerstag','Freitag','Samstag','Sonntag']), 3)
        # Kombiniere und zeige
        df_diff_new = pd.DataFrame([diff_row], columns=df_last4_diff.columns)
        df_diff_preview = pd.concat([df_last4_diff, df_diff_new], ignore_index=True)
        zeige_tabelle(f3, df_diff_preview)

                # prozent_durchschnitt-Tab
        f4 = ttk.Frame(nb); nb.add(f4, text='prozent_durchschnitt')
        df_pct_avg = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prozent_durchschnitt.csv'), engine='python').dropna(how='all')
        try:
            df_pct_avg['Woche_num'] = pd.to_numeric(df_pct_avg['Woche'], errors='coerce')
            df_pct_avg.sort_values('Woche_num', inplace=True)
        except:
            df_pct_avg.sort_values('Woche', inplace=True)
        df_last4_pct = df_pct_avg.tail(4).copy()
        if 'Woche_num' in df_last4_pct.columns:
            df_last4_pct.drop(columns=['Woche_num'], inplace=True)
        # Berechne neue Prozent-Durchschnitte
        df_new = berechne_naechste_woche()
        neue_pct = {}
        for tag in ['Montag','Dienstag','Mittwoch','Donnerstag','Freitag','Samstag','Sonntag']:
            # historische Prozentwerte aus der Tabelle
            hist = pd.to_numeric(df_pct_avg[tag], errors='coerce').fillna(0).tolist()
            # frischer Prozentwert aus neuzugerechneter Wochentage-Zeile
            hist.append(df_new.loc[0, f'proz_{tag.lower()}'])
            neue_pct[tag] = round(sum(hist) / len(hist), 3)
        last_wk_pct = df_last4_pct['Woche'].astype(int).max()
        new_pct_row = {'Woche': last_wk_pct + 1}
        new_pct_row.update(neue_pct)
        df_pct_new = pd.DataFrame([new_pct_row], columns=df_last4_pct.columns)
        df_pct_preview = pd.concat([df_last4_pct, df_pct_new], ignore_index=True)
        zeige_tabelle(f4, df_pct_preview)

        # prozent_unterschied-Tab
        f5 = ttk.Frame(nb); nb.add(f5, text='prozent_unterschied')
        df_pct_diff = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prozent_unterschied.csv'), engine='python').dropna(how='all')
        try:
            df_pct_diff['Woche_num'] = pd.to_numeric(df_pct_diff['Woche'], errors='coerce')
            df_pct_diff.sort_values('Woche_num', inplace=True)
        except:
            df_pct_diff.sort_values('Woche', inplace=True)
        df_last4_diff_pct = df_pct_diff.tail(4).copy()
        if 'Woche_num' in df_last4_diff_pct.columns:
            df_last4_diff_pct.drop(columns=['Woche_num'], inplace=True)
        # Berechne neue Prozent-Unterschiede
        prev_pct = df_last4_pct.iloc[-1]
        diff_pct = {'Woche': new_pct_row['Woche']}
        for tag in ['Montag','Dienstag','Mittwoch','Donnerstag','Freitag','Samstag','Sonntag']:
            diff_pct[tag] = round(new_pct_row[tag] - prev_pct[tag], 3)
        df_pct_diff_new = pd.DataFrame([diff_pct], columns=df_last4_diff_pct.columns)
        df_pct_diff_preview = pd.concat([df_last4_diff_pct, df_pct_diff_new], ignore_index=True)
        zeige_tabelle(f5, df_pct_diff_preview)

        # Buttons unten
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill='x', pady=10)
        ttk.Button(btn_frame, text='Daten übernehmen', command=lambda: print('Daten übernommen')).pack(side='left', padx=5)
        ttk.Button(btn_frame, text='Abbrechen', command=win.destroy).pack(side='left', padx=5)

    ttk.Button(input_frame, text='Vorschau anzeigen', command=open_vorschau).pack(pady=10)
    # 3) Analyse-Tab
    ana_frame = ttk.Frame(main_nb); main_nb.add(ana_frame, text='Analyse'); zeige_analyse(ana_frame)
    root.mainloop()
