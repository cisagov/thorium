# Modifiers

A modifier is a file that tells Thorium how it should modify the remaining stages
of the current reaction. It allows you to change stages arguments and add or remove
tags for this reaction.

A reaction declaration looks something like this:
```json
{
    "args": {
      "adopt": {
        "positionals": ["1"],
        "kwargs": {
          "--color": "black",
        },
        "remove_kwargs": ["--age"],
        "add_switches": ["--puppy"],
        "remove_switches": ["--shedding"],
      }
    },
    "add_tags": ["new_tag"],
    "remove_tags": ["remove_tag"]
}
```

Explanations for the root fields are:

| key | definition |
| --- | ---------- |
| args | The reaction argument update structure (optional) |
| add_tags | The tags to add to this reaction |
| remove_tags | The tags to remove from this reaction |

The argument update structure is similar to the normal arguments structure but has
slightly different fields.

| key | definition |
| --- | ---------- |
| positionals | The positional args to use in place of the original positional args (optional) |
| kwargs | The key word arguments to overlay on top the original keyword arguments |
| remove_kwargs | The key word arguments to remove from the original keyword arguments |
| add_switches | The new switches to add to the original switch arguments |
| remove_switches | The switch arguments to remove from the original switches |

Positional arguments replace the original positionals in their entirety and are not additive.
