# Delete a Group

Any group owner or Thorium admin can delete a group. It is important to remember that *groups* are the resource owners rather than the users that created those resources. Files, repositories, images, pipelines, tool results, and file comments can all be owned by groups. If you delete a group, some of these resources will automatically be purged by the API. For performance reasons, some resources will not be cleaned up when the API services a group deletion request. The following table indicates which resources are cleaned up automatically:

| Group Resources | Removed Upon Group Deletion |
| --- | ---------- |
| Files | No|
| File Comments | No |
| Images | Yes |
| Pipelines | Yes |
| Tool Results | No |
| Reactions | Yes |
| Reaction Stage Logs | No, aged out automatically |
| Repositories | No |

### Archiving Groups Instead
---
Generally we recommend archiving rather than deleting groups. You can do this by adding the Thorium system user to the group as an owner (since groups must have at least 1 owner) and then removing non-admin group members. This preserves data and analysis artifacts without orphaning data and mitigates the risk of future data leakage if that group name was reused by another team. 


### Preparing For Group Deletion
---
If you do want to delete a group, you will need to manually delete any files, repositories, tool results, and file comments using the Web UI, Thorctl, or direct API requests. The following support table details what interfaces support deleting resources:

| Group Resources | Thorctl Deletion | Web UI Deletion | API Deletion Route |
| --- | ---------- | ---------- | ---------- |
| Files | Yes | Yes | Yes |
| File Comments | No | No | No, support planned |
| Images | No | Yes | Yes |
| Pipelines | No | Yes | Yes |
| Tool Results | No | No | Yes |
| Reactions | Yes | Yes | Yes |
| Reaction Stage Logs | No | Yes, delete reaction | Yes |
| Repositories | No | No | Yes |

### Manually Deleting Files
---
When you request to delete a file, you are deleting a file submission from a database. A file can have many different submissions from one or more groups. Therefore, a file will only be deleted from the backend object store when the last submission for it is deleted. This means that a file can be safely "deleted" from one group without removing that file from other groups.

File submissions can be deleted in Thorctl, the Web UI, or through direct API requests. When using Thorctl to delete files in bulk it is important to specify a group to limit the deletion operation to using the `-g` flag. You must also use the `--force` flag when not limiting the deletion to a specific target sha256/tag, because this is considered an especially dangerous operation.

**DANGER: always specify a group using the `-g` flag; otherwise, you may delete files indiscriminately.**

```bash
$ thorctl files delete -g demo-group1234 --force
SHA256                                                           | SUBMISSION                          
-----------------------------------------------------------------+--------------------------------------
3d95783f81e84591dfe8a412c8cec2f5cfcbcbc45ede845bd72b32469e16a34b | 49e8a48b-8ba6-427c-96a9-02a4a9e5ff78 |
...
```
