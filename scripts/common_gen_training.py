#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2018, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import os
import sys
import time

class Options:

    def __init__(self, taskfile_template = '%s-training.tfm', usage = None):
        self.taskfile_template = taskfile_template
        self.usage = usage
        self.set_defaults()
        self.read_options()
        self.make_adjustments()
        self.print_notifications()
        self.check_requirements()

    def set_defaults(self):
        if 'PRJ_DIR' in os.environ:
            self.prj_dir = os.environ['PRJ_DIR']
        else:
            self.prj_dir = '%s/tbemb/tbev-prediction' %os.environ['HOME']
        self.epochs = 60
        self.parser = 'uuparser'
        self.deadline = time.time() + 6.5 * 24 * 3600
        self.first_seed = 300
        self.n_seeds = 9
        self.is_default_deadline = True
        self.is_default_epochs = True
        self.taskfile = None
        self.treebank_folder = '.'
        self.verbose = True
        self.debug = False
        self.help = False

    def read_options(self):
        while len(sys.argv) >= 2 and sys.argv[1].startswith('-'):
            self.last_option_name = sys.argv[1].strip('-').replace('-', '_')
            del sys.argv[1]
            try:
                option = getattr(self, 'option_' + self.last_option_name)
            except:
                raise ValueError('unknown option --%s' %self.last_option_name)
            option()

    def make_adjustments(self):
        if not self.taskfile:
            self.taskfile = self.taskfile_template %self.parser
        self.script = '%s/scripts/run_%s_multi.sh' %(self.prj_dir, self.parser)
        self.seeds = range(self.first_seed, self.first_seed + self.n_seeds)

    def print_notifications(self):
        if self.is_default_deadline and self.deadline and self.verbose:
            # only print this when the user did not specify a deadline
            sys.stderr.write('Tasks will be written with a deadline at %s\n' %time.ctime(self.deadline))
        if self.debug:
            # print all attributes:
            for key in sorted(list(self.__dict__.keys())):
                sys.stderr.write('Option %s:\t%r\n' %(key, getattr(self, key)))
        if self.help:
            if self.usage:
                sys.stdout.write(self.usage)
            else:
                print('Sorry, no help text defined. Please check source.')

    def check_requirements(self):
        if self.help:
            sys.exit()
        if len(sys.argv) > 1:
            raise ValueError('unsupported extra args')
        if os.path.exists(self.taskfile):
            raise ValueError('refusing to overwrite taskfile; remove first')

    def option_deadline(self):
        days = float(self.get_one_arg())
        self.deadline = time.time() + days * 24 * 3600
        self.is_default_deadline = False

    def option_debug(self):
        self.debug = True
        self.verbose = True

    def option_epochs(self):
        self.epochs = int(self.get_one_arg())
        self.is_default_epochs = False

    def option_first_seed(self):
        self.first_seed = int(self.get_one_arg())

    def option_n_seeds(self):
        self.n_seeds = int(self.get_one_arg())

    def option_parser(self): 
        self.parser = self.get_one_arg()
        if self.parser == 'mbist':
            if self.is_default_epochs:
                self.epochs = int(self.epochs/3)
            self.deadline = 0
            self.is_default_deadline = False
        elif self.parser == 'uuparser':
            # note we are not resetting epochs an deadline
            pass
        else:
            raise ValueError, 'unsupported parser %r' %self.parser

    def option_prj_dir(self):
        self.prj_dir = self.get_one_arg()

    def option_quiet(self):
        self.verbose = False

    def option_seeds(self):
        self.options_n_seeds()

    def option_start_seed(self):
        self.option_first_seed()

    def option_taskfile_template(self):
        self.taskfile_template = self.get_one_arg()

    def option_treebank_folder(self):
	self.treebank_folder = self.get_one_arg()

    def option_verbose(self):
        self.verbose = True

    def get_one_arg(self):
        if len(sys.argv) < 2:
            raise ValueError('missing arg for option --%s' %self.last_option_name)
        retval = sys.argv[1]
        del sys.argv[1]
        return retval
        
