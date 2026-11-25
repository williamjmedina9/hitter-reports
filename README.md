# Hitter Reports

Automated hitter reports built from TrackMan data. The script creates cumulative reports for any hitter who appears in the file in a matter of minutes. Provides platoon splits, pitch type splits, spray charts, batted ball quality, and swing decision information.

![Sample hitter report](images/sample-report.png)

Each page shows:
- **Overall Stats:** AVG, OBP, SLG, OPS, wOBA and HR by pitcher handedness
- **Spray Chart** GB / LD / FB / PU, pull / middle / oppo visualized on a spray chart and described in a table
- **Contact Quality:** Exit velocities and hard-hit rate accompanied by a zone showing damage by location
- **Swing Decisions:** BB%, K%, chase rate, and in-zone swing rate accompanied by a zone showing whiff by location
- **Pitch Type Splits:** Performance against fastballs, breaking balls and offspeed pitches from each hand

Switch hitters get a separate page for each side of the plate.

## How to run

Use:

```
python hitter_report.py
```

Script is written for Google Colab:

1. Make sure all files you want to be included in the cumulative report are saved locally on your device.
2. Open a new Colab notebook and paste in the script.
3. Upload the folder containing the CSVs to the session.
4. Change the path that the script uses to access the files.
5. Run the cell. The PDF downloads when it finishes

This will produce a PDF, one page per hitter. The player with the most plate appearances will populate first.

**Requires:** pandas, numpy, matplotlib, Pillow

*Player names in this public version have been removed. Mid-major division one averages gathered from 2026 TrackMan data from the Big West, Mountain West, and West Coast Conferences.*
