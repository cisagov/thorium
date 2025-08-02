export type Group = {
  name: string; // group name
  owners: {
    combined: string[];
    direct: string[];
    metagroups: string[];
  };
  managers: {
    combined: string[];
    direct: string[];
    metagroups: string[];
  };
  analysts: string[];
  users: {
    combined: string[];
    direct: string[];
    metagroups: string[];
  };
  monitors: {
    combined: string[];
    direct: string[];
    metagroups: string[];
  };
  allowed: {
    [resource: string]: boolean;
  };
  description?: string;
};
