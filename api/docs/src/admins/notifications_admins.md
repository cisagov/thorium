# Create Notifications

> This is documentation for Thorium admins looking to manually create notifications.
> For general information on notifications in Thorium, see [Notifications](../developers/notifications.md).

Thorium **Notifications** are usually generated automatically by the Thorium system to communicate important
information to users – for example, that their image or pipeline is banned –, but they can also be created by
Thorium admins manually. This gives admins a mechanism to easily alert users who use/develop a particular
Thorium entity.

## Creating a Notification

You can add a notification to an entity with Thorctl by using the entity's respective subcommand and invoking their
`notifications create` function.

```Bash
thorctl <ENTITY-TYPE> notifications create <group> <ENTITY-NAME> --msg <MESSAGE>
```

### Notification Level

By default, the added notification will have the `INFO` level, but you can manually specify the level as well:

```Bash
... notifications create --level <info/warn/error>
```

### Tying to an Existing Ban

If you want to tie the notification to a particular ban, you can provide the ban's ID. Tying a notification to
a ban will set it to be automatically deleted when the ban is removed.

```Bash
... notifications create ... --ban-id <BAN_ID>
```

### Expiration Behavior

By default, notifications at the `ERROR` level will never "expire" (be deleted automatically), while those on the
`WARN` and `INFO` levels will expire according to the retention settings in the Thorium cluster config (in 7 days
by default). You can set whether notification should automatically expire with the `--expire` flag:

```Bash
... notifications create ... --expire <true/false>
```

## Deleting a Notification

To remove a notification, you'll need to know its ID. You can view notifications' ID's by using the
`--ids/-i` flag with `notifications get`:

```Bash
thorctl <ENTITY-TYPE> notifications get -ids <group> <ENTITY-NAME>
```

This will print the notification ID's along with their contents. Take note of a notification's ID,
then provide it to `notifications delete` to delete it:

```Bash
thorctl <ENTITY-TYPE> notifications delete <ID>
```
