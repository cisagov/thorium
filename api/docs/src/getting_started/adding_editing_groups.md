# Creating/Editing Groups

All resources within Thorium including files, analysis pipelines, and tools are owned by a user and uploaded to a
group. Access to modify group resources is granted by your role within the group. If you want to learn more about
Thorium's roles and permissions system, you can read [this](./adding_editing_groups.md) page. The ability to manage
group membership and create new groups is only available in Thorium's Web UI.

## WebUI
---
To create a new group follow the steps in the following video:

<video autoplay loop controls>
  <source src="../static_resources/create-group.mp4", type="video/mp4">
</video>

You may have noticed that you can add users to different group roles. As we described in the previous chapter, group roles
are how you define the abilities that a group member has within the group. Roles and their abilities are defined in the table
below. Group resources can include images, pipelines, repos, files, tags, comments, and tool results.

| Ability | Owners | Managers | Users | Monitors |
| ---- | ---- | ---- | ---- | ---- |
| View Resources | yes | yes | yes | yes |
| Run Pipelines | yes | yes | yes | no |
| Upload/Create Resources | yes | yes | yes| no |
| Modify/Delete Resources | all | all | self owned only | none |
| Group Membership | add/remove any member | add/remove non-owner members | no | no |
| Delete Group | yes | no | no | no |

A Thorium user can be added to a group either as a direct user or as part of a metagroup. This functionality allows you to use
an external group membership system (ie LDAP/IDM) to grant access to Thorium resources.

| type | description |
| ---- | ----------- |
| direct user | A single user in Thorium |
| metagroups | A group of users that is defined in LDAP |

By default, metagroup info is updated every 10 minutes or when a Thorium group is updated. This means that when a user
is added or removed from a metagroup it may take up to 10 minutes for that change to be visible in Thorium via the Web
UI.