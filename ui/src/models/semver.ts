export type SemVer = {
  major: number;
  minor: number;
  patch: number;
  pre: string; // e.g. "alpha.1" (empty string if none)
  build: string; // e.g. "build.5" (empty string if none)
};
