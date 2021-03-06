from helper.utils import set_seed
from evolve import run_evolve
from evaluate_agent_performance import run_eval_agents
from make_animation import run_make_animation


if __name__ == '__main__':
    set_seed()
    run_evolve()
    run_eval_agents()
    run_make_animation()
