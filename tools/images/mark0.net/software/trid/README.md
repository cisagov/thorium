## Output Explanation

Given a result such as `42.6% (.EXE) UPX compressed Win32 Executable (30569/9/7)`

Breakdown:
1. Percentage is likelyhood of match
2. First number (30569) is the number of points that the corresponding filetype scored against the file
3. The last two point (9, 7) are used by the developer for debugging the program. As the user, we should not be concerned with them.

Source:
https://mark0.net/forum/index.php?topic=56.0

## Other Documentation
`chmod a+x trid` is needed to allow trid to run

`export LC_ALL=C` is needed to prevent trid from crashing
