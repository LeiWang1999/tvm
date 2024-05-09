# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The core tuning API"""
from typing import List, Optional

from .builder import Builder
from .cost_model import CostModel
from .database import Database
from .measure_callback import MeasureCallback
from .runner import Runner
from .task_scheduler import TaskScheduler
from .tune_context import TuneContext
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

from . import default_config
# from .apply_history_best import ApplyHistoryBest
from .builder import Builder
from .cost_model import CostModel
from .database import Database, TuningRecord
from .extracted_task import ExtractedTask
from .measure_callback import MeasureCallback
from .mutator import Mutator
from .postproc import Postproc
from .runner import Runner
from .schedule_rule import ScheduleRule
from .search_strategy import EvolutionarySearch, ReplayFunc, ReplayTrace
from .space_generator import SpaceGenerator
from .task_scheduler import GradientBased, RoundRobin
from .tune_context import TuneContext
from .utils import batch_parameterize_config

FnSpaceGenerator = Callable[[], SpaceGenerator]
FnScheduleRule = Callable[[], List[ScheduleRule]]
FnPostproc = Callable[[], List[Postproc]]
FnMutatorProb = Callable[[], Dict[Mutator, float]]

class TuneConfig(NamedTuple):
    """Configuration for tuning

    Parameters
    ----------
    max_trials_global: int
        Maximum number of trials to run.
    num_trials_per_iter: int
        Number of trials to run per iteration.
    max_trials_per_task: Optional[int]
        Maximum number of trials to run per task. If None, use `max_trials_global`.
    task_scheduler: str = "gradient"
        Task scheduler to use.
        Valid options are: round_robin, gradient.
    strategy: str = "evolutionary"
        Search strategy to use.
        Valid options are: evolutionary, replay_func, replay_trace.
    task_scheduler_config: Optional[Dict[str, Any]] = None
        Configuration for task scheduler.
    search_strategy_config: Optional[Dict[str, Any]] = None
        Configuration for search strategy.
    logger_config: Optional[Dict[str, Any]] = None
        Configuration for logger.
    """

    max_trials_global: int
    num_trials_per_iter: int
    max_trials_per_task: Optional[int] = None
    task_scheduler: str = "gradient"
    strategy: str = "evolutionary"
    task_scheduler_config: Optional[Dict[str, Any]] = None
    search_strategy_config: Optional[Dict[str, Any]] = None
    logger_config: Optional[Dict[str, Any]] = None

    def create_strategy(self, **kwargs):
        """Create search strategy from configuration"""
        cls_tbl = {
            "evolutionary": EvolutionarySearch,
            "replay_func": ReplayFunc,
            "replay_trace": ReplayTrace,
        }
        if self.strategy not in cls_tbl:
            raise ValueError(
                f"Invalid search strategy: {self.strategy}. "
                "Valid options are: {}".format(", ".join(cls_tbl.keys()))
            )
        # `max_trials_per_task` defaults to `max_trials_global`
        max_trials_per_task = self.max_trials_per_task
        if max_trials_per_task is None:
            max_trials_per_task = self.max_trials_global
        # `search_strategy_config` defaults to empty dict
        config = self.search_strategy_config
        if config is None:
            config = {}
        return cls_tbl[self.strategy](
            num_trials_per_iter=self.num_trials_per_iter,
            max_trials_per_task=max_trials_per_task,
            **kwargs,
            **config,
        )

    def create_task_scheduler(self, **kwargs):
        """Create task scheduler from configuration"""
        cls_tbl = {
            "round_robin": RoundRobin,
            "gradient": GradientBased,
        }
        if self.task_scheduler not in cls_tbl:
            raise ValueError(
                f"Invalid task scheduler: {self.task_scheduler}. "
                "Valid options are: {}".format(", ".join(cls_tbl.keys()))
            )
        # `task_scheduler_config` defaults to empty dict
        config = self.task_scheduler_config
        if config is None:
            config = {}
        return cls_tbl[self.task_scheduler](
            max_trials=self.max_trials_global,
            **kwargs,
            **config,
        )

    def create_loggers(
        self,
        log_dir: str,
        params: List[Dict[str, Any]],
        disable_existing_loggers: bool = False,
    ):
        """Create loggers from configuration"""
        if self.logger_config is None:
            config = {}
        else:
            config = self.logger_config

        config.setdefault("loggers", {})
        config.setdefault("handlers", {})
        config.setdefault("formatters", {})

        global_logger_name = "tvm.meta_schedule"
        global_logger = logging.getLogger(global_logger_name)
        if global_logger.level is logging.NOTSET:
            global_logger.setLevel(logging.INFO)

        config["loggers"].setdefault(
            global_logger_name,
            {
                "level": logging._levelToName[  # pylint: disable=protected-access
                    global_logger.level
                ],
                "handlers": [handler.get_name() for handler in global_logger.handlers]
                + [global_logger_name + ".console", global_logger_name + ".file"],
                "propagate": False,
            },
        )
        config["loggers"].setdefault(
            "{logger_name}",
            {
                "level": "INFO",
                "handlers": [
                    "{logger_name}.file",
                ],
                "propagate": False,
            },
        )
        config["handlers"].setdefault(
            global_logger_name + ".console",
            {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "tvm.meta_schedule.standard_formatter",
            },
        )
        config["handlers"].setdefault(
            global_logger_name + ".file",
            {
                "class": "logging.FileHandler",
                "filename": "{log_dir}/" + __name__ + ".task_scheduler.log",
                "mode": "a",
                "level": "INFO",
                "formatter": "tvm.meta_schedule.standard_formatter",
            },
        )
        config["handlers"].setdefault(
            "{logger_name}.file",
            {
                "class": "logging.FileHandler",
                "filename": "{log_dir}/{logger_name}.log",
                "mode": "a",
                "level": "INFO",
                "formatter": "tvm.meta_schedule.standard_formatter",
            },
        )
        config["formatters"].setdefault(
            "tvm.meta_schedule.standard_formatter",
            {
                "format": "%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        )

        # set up dictConfig loggers
        p_config = {"version": 1, "disable_existing_loggers": disable_existing_loggers}
        for k, v in config.items():
            if k in ["formatters", "handlers", "loggers"]:
                p_config[k] = batch_parameterize_config(v, params)  # type: ignore
            else:
                p_config[k] = v
        logging.config.dictConfig(p_config)

        # check global logger
        if global_logger.level not in [logging.DEBUG, logging.INFO]:
            global_logger.warning(
                "Logging level set to %s, please set to logging.INFO"
                " or logging.DEBUG to view full log.",
                logging._levelToName[global_logger.level],  # pylint: disable=protected-access
            )
        global_logger.info("Logging directory: %s", log_dir)


def tune_extracted_tasks(
    extracted_tasks: List[ExtractedTask],
    config: TuneConfig,
    work_dir: str,
    *,
    builder: Optional[Builder] = None,
    runner: Optional[Runner] = None,
    database: Optional[Database] = None,
    cost_model: Optional[CostModel] = None,
    measure_callbacks: Optional[List[MeasureCallback]] = None,
    space: Optional[FnSpaceGenerator] = None,
    sch_rules: Optional[FnScheduleRule] = None,
    postprocs: Optional[FnPostproc] = None,
    mutator_probs: Optional[FnMutatorProb] = None,
    num_threads: Optional[int] = None,
) -> Database:
    """Tune extracted tasks with a given target.

    Parameters
    ----------
    extracted_tasks : List[ExtractedTask]
        The list of extracted tasks.
    config : TuneConfig
        The search strategy config.
    work_dir : Optional[str]
        The working directory to save intermediate results.
    builder : Optional[Builder]
        The builder to use.
    runner : Optional[Runner]
        The runner to use.
    database : Optional[Database]
        The database to use.
    cost_model : Optional[CostModel]
        The cost model to use.
    measure_callbacks : Optional[List[MeasureCallback]]
        The callbacks used during tuning.
    task_scheduler : Optional[TaskScheduler]
        The task scheduler to use.
    space : Optional[FnSpaceGenerator]
        The space generator to use.
    sch_rules : Optional[FnScheduleRule]
        The search rules to use.
    postprocs : Optional[FnPostproc]
        The postprocessors to use.
    mutator_probs : Optional[FnMutatorProb]
        The probability distribution to use different mutators.
    num_threads : Optional[int]
        The number of threads to use.

    Returns
    -------
    database : Database
        The database containing all the tuning results.

    """
    # pylint: disable=protected-access
    # logging directory is set to `work_dir/logs` by default
    log_dir = osp.join(work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    max_width = len(str(len(extracted_tasks) - 1))
    logger_name_pattern = __name__ + ".task_{task_id:0" + f"{max_width}" + "d}_{task_name}"

    config.create_loggers(
        log_dir=log_dir,
        params=[
            {
                "log_dir": log_dir,
                "logger_name": logger_name_pattern.format(task_id=i, task_name=task.task_name),
            }
            for i, task in enumerate(extracted_tasks)
        ],
    )

    logger.info("Working directory: %s", work_dir)
    database = default_config.database(database, work_dir)
    builder = default_config.builder(builder)
    runner = default_config.runner(runner)
    cost_model = default_config.cost_model(cost_model)
    measure_callbacks = default_config.callbacks(measure_callbacks)
    # parse the tuning contexts
    tune_contexts = []
    for i, task in enumerate(extracted_tasks):
        assert len(task.dispatched) == 1, "Only size 1 dispatched task list is supported for now"
        tune_contexts.append(
            TuneContext(
                mod=default_config.mod(task.dispatched[0]),
                target=task.target,
                space_generator=default_config.space_generator(space),
                search_strategy=config.create_strategy(),
                sch_rules=default_config.schedule_rules(sch_rules, task.target),
                postprocs=default_config.postproc(postprocs, task.target),
                mutator_probs=default_config.mutator_probs(mutator_probs, task.target),
                task_name=task.task_name,
                logger=logging.getLogger(
                    logger_name_pattern.format(task_id=i, task_name=task.task_name)
                ),
                num_threads=num_threads,
            )
        )
    # parse the task scheduler
    # pylint: enable=protected-access
    task_scheduler = config.create_task_scheduler(
        tasks=tune_contexts,
        task_weights=[float(t.weight) for t in extracted_tasks],
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
    )
    task_scheduler.tune()
    cost_model.save(osp.join(work_dir, "cost_model.xgb"))
    return database



def tune_tasks(
    *,
    tasks: List[TuneContext],
    task_weights: List[float],
    work_dir: str,
    max_trials_global: int,
    max_trials_per_task: Optional[int] = None,
    num_trials_per_iter: int = 64,
    builder: Builder.BuilderType = "local",
    runner: Runner.RunnerType = "local",
    database: Database.DatabaseType = "json",
    cost_model: CostModel.CostModelType = "xgb",
    measure_callbacks: MeasureCallback.CallbackListType = "default",
    task_scheduler: TaskScheduler.TaskSchedulerType = "gradient",
    module_equality: str = "structural",
) -> Database:
    """Tune a list of tasks. Using a task scheduler.

    Parameters
    ----------
    tasks : List[TuneContext]
        The list of tasks to tune.
    task_weights : List[float]
        The weight of each task.
    work_dir : str
        The working directory.
    max_trials_global : int
        The maximum number of trials to run globally.
    max_trials_per_task : Optional[int]
        The maximum number of trials to run per task.
    num_trials_per_iter : int
        The number of trials to run per iteration
    builder : Builder.BuilderType
        The builder.
    runner : Runner.RunnerType
        The runner.
    database : Database.DatabaseType
        The database.
    cost_model : CostModel.CostModelType
        The cost model.
    measure_callbacks : MeasureCallback.CallbackListType
        The measure callbacks.
    task_scheduler : TaskScheduler.TaskSchedulerType
        The task scheduler.
    module_equality : Optional[str]
        A string to specify the module equality testing and hashing method.
        It must be one of the followings:

            - "structural": Use StructuralEqual/Hash
            - "ignore-ndarray": Same as "structural", but ignore ndarray raw data during equality
                testing and hashing.
            - "anchor-block": Apply equality testing and hashing on the anchor block extracted from
                a given module. The "ignore-ndarray" varint is used for the extracted blocks or in
                case no anchor block is found. For the definition of the anchor block, see
                tir/analysis/analysis.py.

    Returns
    -------
    database : Database
        The database with all tuning records
    """
    if len(tasks) == 0:
        raise ValueError("No tasks to tune.")

    if len(tasks) != len(task_weights):
        raise ValueError(
            f"Length of tasks ({len(tasks)}) and task_weights ({len(task_weights)}) do not match."
        )

    num_cores = tasks[0].num_threads

    if max_trials_per_task is None:
        max_trials_per_task = max_trials_global
    if not isinstance(builder, Builder):
        builder = Builder.create(builder, max_workers=num_cores)
    if not isinstance(runner, Runner):
        runner = Runner.create(runner, max_workers=num_cores)
    if database == "json":
        database = Database.create(database, work_dir=work_dir, module_equality=module_equality)
    elif not isinstance(database, Database):
        database = Database.create(database, module_equality=module_equality)
    if not isinstance(cost_model, CostModel):
        cost_model = CostModel.create(cost_model, num_tuning_cores=num_cores, tree_method="auto")
    if isinstance(measure_callbacks, MeasureCallback):
        measure_callbacks = [measure_callbacks]
    elif measure_callbacks == "default":
        measure_callbacks = MeasureCallback.create(measure_callbacks)
    if not isinstance(task_scheduler, TaskScheduler):
        task_scheduler = TaskScheduler.create(task_scheduler)
    task_scheduler.tune(
        tasks=tasks,
        task_weights=task_weights,
        max_trials_global=max_trials_global,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        builder=builder,
        runner=runner,
        measure_callbacks=measure_callbacks,
        database=database,
        cost_model=cost_model,
    )
    return database
