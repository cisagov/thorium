Whats changed in Thorium!
## [unreleased]

### 🚀 Features

- *(agent)* Added support for restricting children to specific groups - (3614835)
- *(thorctl)* Added support for repos in thorctl run - (6014c46)
- *(thorctl)* Added support for listing repo commitishes - (c760d61)
- *(thorctl)* Added support for pruning downloaded repos - (3ae76b3)
- *(thorctl)* Added support downloading repos from a file - (f6580c3)
- *(thorctl)* Added support for showing progress while downloading repos - (a5d47f0)
- *(thorctl)* Added support skipping already downloaded repos - (98f5b51)
- *(thorctl)* Added thorctl base image - (4cc28ed)
- *(thorctl)* [**breaking**] Added support for multi-pipeline reaction creation - (4673429)
- *(ui)* Added favicon support - (e907abb)

### 🐛 Bug Fixes

- *(api)* Fixed issue where api where suppress 404 errors - (34681b4)
- *(api)* Resolved issue with worker tombstones - (6e12f9a)
- *(thorctl)* Changed describe to 'pretty' mode by default - (bf47a5f)
- *(thorctl)* Swapped positions of commitish and kind - (35c7e6e)
- *(thorctl)* Changed reaction logs to print to stdout by default - (c9f50fd)
- *(thorctl)* Fixed 404 when getting repo run results with a commitish - (5c670f1)
- *(thorctl)* Fixed issue where repos with '.' in the name were incorrectly ingested - (b851262)
- *(thorctl)* Added support for updating repos with no add groups - (9fbc805)
- *(thorctl)* Corrected Thorctl results handling - (b69a004)
- *(thorctl)* Changed Thorctl login to keep settings by default - (7ef3c97)
- *(ui)* Fixed issue where the related tab had a strange unicode character - (da46c28)

### 📚 Documentation

- *(agent)* Documented resolving issues with jobs stuck in a created state - (9532c1b)
- *(api)* Documented the different between CaRT and Encrypted Zips - (24d241c)
- *(api)* Improved docs for file upload - (86b5ebb)
- *(scaler)* Added documentation for scheduling pools - (73a7aba)
- *(thorctl)* Updated documentation on thorctl login - (93f2b34)

## [1.102.0] - 2024-08-29

### 🚀 Features

- *(!thorctl)* Added support for retrying cursors on server side errors - (a77fb8a)
- *(api)* Added support for restricting registration by email - (542bae3)
- *(api)* Added support for basic reasoning in reset logs - (2fc1989)
- *(thorctl)* Parse simple keys file in lieu of config - (68f64ba)
- *(thorctl)* Thorctl will automatically update configs to the new api url - (b33083f)
- *(thorctl)* Added support for organized repo downloads - (157a5a4)
- *(thorctl)* Added support for creating sub reactions - (efdd549)

### 🐛 Bug Fixes

- *(agent)* Fixed issue where agent would fail to uncart files - (0adbb65)
- *(agent)* Fixed issue where agents didn't properly enforce fair share lifetimes - (10334d3)
- *(agent)* Fixed issue where runtime lifetimes were being incorrectly checked - (05cff4f)
- *(api)* Fixed issue where listing repo tags would sometimes panic - (0354675)
- *(api)* Fixed issue where jobs could end up in a dangling state - (2c40cf0)
- *(api)* Fixed issue where cursors ties would cause panics - (1b4e171)
- *(api)* Fixed issue where nonexistent users could be added to groups - (22ece6b)
- *(api)* Fixed issue where deleting reactions didn't clean up the running queue - (7926f91)
- *(api)* Fixed issue where the users wouldn't get redirected after email verification - (0db1a83)
- *(api)* Optimized tag creation when duplicate values are created - (9073b5d)
- *(operator)* Fixed ThoriumCluster finalization deadlock - (18a7724)
- *(scaler)* Fixed issue where the scaler would needlessly downscale workers - (f1fff3f)
- *(thorctl)* Fix Windows interactive login - (a352681)
- *(thorctl)* Fixed issue where listing results can run out of file descriptors - (6dafeec)

### 📚 Documentation

- *(api)* Add docs explaining tag dependencies - (1c4037b)

