//! Set up the correct cgroups for a specific image
use cgroups_rs::cpu::CpuController;
use cgroups_rs::memory::MemController;
use cgroups_rs::Controller;
use cgroups_rs::{cgroup_builder::CgroupBuilder, CgroupPid};
use thorium::models::Image;
use thorium::Error;

/// a specific cgroup for a specific image
pub struct Cgroup {
    /// The cgroup we will add new processes too
    cgroup: cgroups_rs::Cgroup,
}

impl Cgroup {
    /// Create a new cgroup
    #[rustfmt::skip]
    pub fn new(name: &str, image: &Image) -> Result<Self, Error> {
        //get a handle to the cgroup hierarchy
        let hierarchy = cgroups_rs::hierarchies::auto();
        // build our cgroup for this image
        let cgroup = CgroupBuilder::new(&format!("thorium/{}", name))
            .cpu()
            .shares(image.resources.cpu)
            .done()
            .memory()
            .memory_hard_limit((image.resources.memory * 1048576) as i64)
            .done()
            .build(hierarchy)?;
        Ok(Cgroup { cgroup })
    }

    /// Tries to load an existing control group
    pub fn load(name: &str) -> Self {
        //get a handle to the cgroup hierarchy
        let hierarchy = cgroups_rs::hierarchies::auto();
        // build the path to our cgroup
        let path = format!("thorium/{}", name);
        let cgroup = cgroups_rs::Cgroup::load(hierarchy, &path);
        Cgroup { cgroup }
    }

    /// Add a new process to this cgroup
    ///
    /// # Arguments
    ///
    /// * `pid` - The process id to add to this cgroup
    pub fn add(&mut self, pid: u32) -> Result<(), Error> {
        // get the cpu and memory controller for this cgroup
        let cpu_controller: &CpuController = self
            .cgroup
            .controller_of()
            .ok_or_else(|| Error::new("Failed to get cgroups cpu controller"))?;
        let mem_controller: &MemController = self
            .cgroup
            .controller_of()
            .ok_or_else(|| Error::new("Failed to get cgroups memory controller"))?;
        // cast our pid to a group pid
        let cgroup_pid = CgroupPid::from(pid as u64);
        // add this pid to our cpu and memory controllers
        cpu_controller.add_task(&cgroup_pid)?;
        mem_controller.add_task(&cgroup_pid)?;
        Ok(())
    }

    /// Delete a no longer in use cgroup
    pub fn delete(&mut self) -> Result<(), Error> {
        // delete this cgroup
        self.cgroup.delete()?;
        Ok(())
    }

    /// list the processes for this cgroup
    pub fn procs(&self) -> Vec<CgroupPid> {
        // get the procs for this cgroup
        self.cgroup.procs()
    }
}
