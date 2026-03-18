// Returns the total member count including all users,
//   monitors, managers and owners
const GroupMemberCount = ({ group }) => {
  const count =
    group.owners.combined.length + group.users.combined.length + group.managers.combined.length + group.monitors.combined.length;
  // use plural string if multiple members
  if (count > 1) {
    return <>{count} Members</>;
  } else {
    return <>{count} Member</>;
  }
};

export default GroupMemberCount;
